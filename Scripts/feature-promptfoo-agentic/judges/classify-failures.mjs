import fs from 'node:fs/promises';
import path from 'node:path';

function readEnv(name, fallback = undefined) {
  const value = process.env[name];
  return value && value.length > 0 ? value : fallback;
}

function safeJsonParse(value) {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: 'Bearer promptfoo',
    },
    body: JSON.stringify(body),
  });

  const text = await response.text();
  const parsed = safeJsonParse(text);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${text}`);
  }
  if (!parsed) {
    throw new Error(`Judge returned non-JSON response: ${text}`);
  }
  return parsed;
}

function clip(value, limit = 4000) {
  if (value == null) return value;
  const str = typeof value === 'string' ? value : JSON.stringify(value);
  return str.length <= limit ? str : `${str.slice(0, limit)}...<truncated>`;
}

function classifySchema() {
  return {
    name: 'afm_failure_classification',
    schema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        classification: {
          type: 'string',
          enum: ['afm_bug', 'model_quality', 'harness_bug'],
        },
        confidence: {
          type: 'number',
        },
        rationale: {
          type: 'string',
        },
        evidence: {
          type: 'array',
          items: { type: 'string' },
        },
      },
      required: ['classification', 'confidence', 'rationale', 'evidence'],
    },
  };
}

function buildUserPayload(result) {
  return {
    test_description: result.testCase?.description ?? null,
    provider: result.provider?.label ?? null,
    prompt: result.prompt?.raw ?? result.vars?.prompt ?? null,
    assertions: result.testCase?.assert ?? [],
    grading_reason: result.gradingResult?.reason ?? null,
    component_results: result.gradingResult?.componentResults ?? [],
    normalized_output: result.response?.output ?? null,
    request_body: result.response?.metadata?.requestBody ?? null,
    raw_response_body: result.response?.metadata?.responseBody ?? null,
    latency_ms: result.latencyMs ?? null,
  };
}

async function classifyFailure({ result, judgeBaseUrl, judgeModel, systemPrompt }) {
  const payload = buildUserPayload(result);
  const body = {
    model: judgeModel,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: JSON.stringify(payload) },
    ],
    temperature: 0,
    stream: false,
    response_format: {
      type: 'json_schema',
      json_schema: classifySchema(),
    },
  };

  const response = await postJson(`${judgeBaseUrl.replace(/\/+$/, '')}/chat/completions`, body);
  const content = response?.choices?.[0]?.message?.content ?? '';
  const parsed = safeJsonParse(content);
  if (!parsed) {
    throw new Error(`Judge returned non-JSON classification content: ${clip(content, 800)}`);
  }
  return parsed;
}

function augmentReport(report, classifications) {
  const byId = new Map(classifications.map((item) => [item.id, item]));
  const augmented = structuredClone(report);
  for (const result of augmented.results?.results ?? []) {
    const classification = byId.get(result.id);
    if (classification) {
      result.aiClassification = classification;
    }
  }
  return augmented;
}

function buildSummary(reportPath, classifications) {
  const total = classifications.length;
  const afmBug = classifications.filter((c) => c.classification === 'afm_bug').length;
  const modelQuality = classifications.filter((c) => c.classification === 'model_quality').length;
  const harnessBug = classifications.filter((c) => c.classification === 'harness_bug').length;

  const lines = [
    `# AI Failure Classification`,
    ``,
    `Source report: \`${reportPath}\``,
    ``,
    `- failures classified: ${total}`,
    `- afm_bug: ${afmBug}`,
    `- model_quality: ${modelQuality}`,
    `- harness_bug: ${harnessBug}`,
    ``,
    `## Details`,
    ``,
  ];

  for (const item of classifications) {
    lines.push(`### ${item.testDescription || item.id}`);
    lines.push(`- provider: ${item.provider}`);
    lines.push(`- classification: ${item.classification}`);
    lines.push(`- confidence: ${item.confidence}`);
    lines.push(`- rationale: ${item.rationale}`);
    if (item.evidence?.length) {
      lines.push(`- evidence:`);
      for (const evidence of item.evidence) {
        lines.push(`  - ${evidence}`);
      }
    }
    lines.push('');
  }

  return `${lines.join('\n')}\n`;
}

async function main() {
  const reportPath = process.argv[2];
  if (!reportPath) {
    console.error('Usage: node Scripts/feature-promptfoo-agentic/judges/classify-failures.mjs <promptfoo-report.json>');
    process.exit(1);
  }

  const judgeBaseUrl = readEnv('AFM_JUDGE_BASE_URL', 'http://127.0.0.1:9999/v1');
  const judgeModel = readEnv('AFM_JUDGE_MODEL', readEnv('AFM_MODEL'));
  if (!judgeModel) {
    throw new Error('Set AFM_JUDGE_MODEL or AFM_MODEL for the judge model id');
  }

  const promptPath = path.join(
    path.dirname(new URL(import.meta.url).pathname),
    'failure-classifier-prompt.md',
  );
  const systemPrompt = await fs.readFile(promptPath, 'utf8');
  const report = JSON.parse(await fs.readFile(reportPath, 'utf8'));

  const failed = (report.results?.results ?? []).filter((result) => !result.success);
  const classifications = [];
  for (const result of failed) {
    const classified = await classifyFailure({ result, judgeBaseUrl, judgeModel, systemPrompt });
    classifications.push({
      id: result.id,
      testDescription: result.testCase?.description ?? null,
      provider: result.provider?.label ?? null,
      classification: classified.classification,
      confidence: classified.confidence,
      rationale: classified.rationale,
      evidence: classified.evidence,
    });
  }

  const augmented = augmentReport(report, classifications);
  const baseName = reportPath.replace(/\.json$/i, '');
  const classifiedPath = `${baseName}.classified.json`;
  const summaryPath = `${baseName}.classified.summary.md`;

  await fs.writeFile(classifiedPath, JSON.stringify(augmented, null, 2));
  await fs.writeFile(summaryPath, buildSummary(reportPath, classifications));

  console.log(JSON.stringify({
    source: reportPath,
    classifiedReport: classifiedPath,
    summary: summaryPath,
    failuresClassified: classifications.length,
  }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exit(1);
});
