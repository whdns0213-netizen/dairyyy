import 'dotenv/config';
import express, { type Request, type Response } from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { deflateRawSync } from 'zlib';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const MODEL_NAME = 'gemini-4-hydrogen';

const DOWNLOADABLE_SOURCE_FILES = [
  '.env.example',
  '.gitignore',
  'index.html',
  'metadata.json',
  'package.json',
  'server.ts',
  'tsconfig.json',
  'vite.config.ts',
  'src/App.tsx',
  'src/index.css',
  'src/main.tsx',
] as const;

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(buf: Buffer): number {
  let crc = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    crc = CRC_TABLE[(crc ^ buf[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function buildSourceZip(rootDir: string): Buffer {
  const localParts: Buffer[] = [];
  const centralParts: Buffer[] = [];
  let offset = 0;
  let fileCount = 0;

  for (const relPath of DOWNLOADABLE_SOURCE_FILES) {
    const resolved = path.resolve(rootDir, relPath);
    if (!resolved.startsWith(rootDir + path.sep) || !fs.existsSync(resolved)) {
      continue;
    }

    const raw = fs.readFileSync(resolved);
    const compressed = deflateRawSync(raw);
    const checksum = crc32(raw);
    const nameBuf = Buffer.from(relPath, 'utf8');

    const localHeader = Buffer.alloc(30);
    localHeader.writeUInt32LE(0x04034b50, 0);
    localHeader.writeUInt16LE(20, 4);
    localHeader.writeUInt16LE(0x0800, 6);
    localHeader.writeUInt16LE(8, 8);
    localHeader.writeUInt16LE(0, 10);
    localHeader.writeUInt16LE(0x5421, 12);
    localHeader.writeUInt32LE(checksum, 14);
    localHeader.writeUInt32LE(compressed.length, 18);
    localHeader.writeUInt32LE(raw.length, 22);
    localHeader.writeUInt16LE(nameBuf.length, 26);
    localHeader.writeUInt16LE(0, 28);

    localParts.push(localHeader, nameBuf, compressed);

    const centralHeader = Buffer.alloc(46);
    centralHeader.writeUInt32LE(0x02014b50, 0);
    centralHeader.writeUInt16LE(20, 4);
    centralHeader.writeUInt16LE(20, 6);
    centralHeader.writeUInt16LE(0x0800, 8);
    centralHeader.writeUInt16LE(8, 10);
    centralHeader.writeUInt16LE(0, 12);
    centralHeader.writeUInt16LE(0x5421, 14);
    centralHeader.writeUInt32LE(checksum, 16);
    centralHeader.writeUInt32LE(compressed.length, 20);
    centralHeader.writeUInt32LE(raw.length, 24);
    centralHeader.writeUInt16LE(nameBuf.length, 28);
    centralHeader.writeUInt16LE(0, 30);
    centralHeader.writeUInt16LE(0, 32);
    centralHeader.writeUInt16LE(0, 34);
    centralHeader.writeUInt16LE(0, 36);
    centralHeader.writeUInt32LE(0, 38);
    centralHeader.writeUInt32LE(offset, 42);

    centralParts.push(centralHeader, nameBuf);

    offset += localHeader.length + nameBuf.length + compressed.length;
    fileCount += 1;
  }

  const centralDir = Buffer.concat(centralParts);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(0, 4);
  eocd.writeUInt16LE(0, 6);
  eocd.writeUInt16LE(fileCount, 8);
  eocd.writeUInt16LE(fileCount, 10);
  eocd.writeUInt32LE(centralDir.length, 12);
  eocd.writeUInt32LE(offset, 16);
  eocd.writeUInt16LE(0, 20);

  return Buffer.concat([...localParts, centralDir, eocd]);
}

function createGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('MISSING_API_KEY');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

async function generateWithRetry(
  ai: GoogleGenAI,
  diaryContent: string,
  moodLabel: string,
  dateStr: string
): Promise<string> {
  const maxRetries = 2;
  let lastError: unknown;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await ai.models.generateContent({
        model: MODEL_NAME,
        contents: [
          `작성 날짜: ${dateStr || '오늘'}`,
          `선택한 감정: ${moodLabel || '차분함'}`,
          `일기 내용:\n${diaryContent}`,
        ].join('\n'),
        config: {
          systemInstruction:
            '당신은 감성적인 한국어 일기장 "감성일기장"의 다정한 마음 친구입니다. ' +
            '사용자가 작성한 오늘의 일기와 선택한 감정을 읽고, 하루를 따뜻하게 감싸주는 "오늘의 위로 한마디"를 작성해 주세요. ' +
            '규칙: 1~2문장(공백 포함 110자 이내)의 서정적이고 진심 어린 한국어 경어체("~요"체)로 작성하세요. ' +
            '훈계하거나 억지 긍정으로 몰아가지 말고, 일기 속 구체적인 상황이나 감정에 조용히 공감하고 다독여 주세요. ' +
            '따옴표, 해시태그, 마크다운 기호 없이 문장 본문만 출력하세요.',
          thinkingConfig: {
            thinkingLevel: ThinkingLevel.LOW,
          },
        },
      });

      const text = response.text?.trim();
      if (!text) {
        throw new Error('EMPTY_RESPONSE');
      }
      return text;
    } catch (err) {
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);
      const isTransient = msg.includes('429') || msg.includes('503') || msg.includes('UNAVAILABLE');
      if (!isTransient || attempt === maxRetries) {
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 500 * Math.pow(2, attempt)));
    }
  }

  throw lastError;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '100kb' }));

  app.post('/api/comfort', async (req: Request, res: Response) => {
    const body = req.body as { content?: unknown; moodLabel?: unknown; date?: unknown } | undefined;

    if (!body || typeof body.content !== 'string') {
      res.status(400).json({ error: '일기 내용을 문자열로 전달해 주세요.' });
      return;
    }

    const content = body.content.trim();
    if (!content) {
      res.status(400).json({ error: '일기 내용이 비어 있습니다.' });
      return;
    }
    if (content.length > 5000) {
      res.status(400).json({ error: '일기 내용은 5,000자 이내로 입력해 주세요.' });
      return;
    }

    const moodLabel = typeof body.moodLabel === 'string' ? body.moodLabel.trim().slice(0, 40) : '차분함';
    const dateStr = typeof body.date === 'string' ? body.date.trim().slice(0, 20) : '';

    try {
      const ai = createGeminiClient();
      const comfortMessage = await generateWithRetry(ai, content, moodLabel, dateStr);
      res.json({ comfortMessage });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg === 'MISSING_API_KEY' || msg.includes('API_KEY_INVALID') || msg.includes('PERMISSION_DENIED')) {
        res.status(500).json({
          error: 'GEMINI_API_KEY가 설정되지 않았거나 유효하지 않습니다. Settings > Secrets 패널에서 API 키를 확인해 주세요.',
        });
        return;
      }
      if (msg.includes('RESOURCE_EXHAUSTED') || msg.includes('429')) {
        res.status(429).json({
          error: '요청 한도가 초과되었습니다. 잠시 후 다시 시도해 주세요.',
        });
        return;
      }
      console.error('Gemini comfort generation error:', err);
      res.status(502).json({
        error: 'AI 위로 한마디를 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.',
      });
    }
  });

  app.get('/api/download-code', (_req: Request, res: Response) => {
    try {
      const zipBuffer = buildSourceZip(__dirname);
      res.setHeader('Content-Type', 'application/zip');
      res.setHeader('Content-Disposition', 'attachment; filename="mood-vibe-diary-code.zip"');
      res.setHeader('Cache-Control', 'no-store');
      res.send(zipBuffer);
    } catch (err) {
      console.error('Source download error:', err);
      res.status(500).json({ error: '소스코드를 압축하는 중 오류가 발생했습니다.' });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('/{*splat}', (_req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Mood & Vibe Diary server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
