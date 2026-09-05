#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';

const [outDir, model, minimumMtimeArgument] = process.argv.slice(2);
if (!outDir || !model) {
  console.error('Usage: summarize-results.mjs <report-directory> <model> [minimum-mtime-ms]');
  process.exit(1);
}

const minimumMtimeMs = minimumMtimeArgument === undefined
  ? null
  : Number(minimumMtimeArgument);
if (minimumMtimeMs !== null && (!Number.isFinite(minimumMtimeMs) || minimumMtimeMs < 0)) {
  throw new Error(`invalid minimum report mtime: ${minimumMtimeArgument}`);
}

const modelSlug = model.replace(/[/:]/g, '_');
const reportSuffix = `-${modelSlug}.json`;
const outputBase = path.join(outDir, `promptfoo-summary-${modelSlug}`);

const buckets = {
  nativeProtocolConformance: {
    title: 'Native protocol conformance',
    description: 'Structured output, grammar, transport, and native/auto-detected tool-call behavior.',
    files: [],
    successes: 0,
    failures: 0,
    errors: 0,
    uniqueFailingCases: new Set(),
  },
  modelAgentBehaviorQuality: {
    title: 'Model/agent behavior quality',
    description: 'Native-parser semantic tool selection and framework workflow preferences.',
    files: [],
    successes: 0,
    failures: 0,
    errors: 0,
    uniqueFailingCases: new Set(),
  },
  forcedParserCompatibility: {
    title: 'Forced-parser compatibility experiments',
    description: 'Non-native adaptive-XML override experiments; excluded from native conformance totals.',
    files: [],
    successes: 0,
    failures: 0,
    errors: 0,
    uniqueFailingCases: new Set(),
  },
};

const failureCauseLabels = [
  'engine/runtime likely',
  'model behavior likely',
  'test harness',
  'forced-parser experiment',
  'unresolved',
];
const failureTaxonomy = Object.fromEntries(failureCauseLabels.map((label) => [label, {
  count: 0,
  uniqueFailingCases: new Set(),
  records: [],
}]));

const nativeFilename = /^(structured|structured-stress|toolcall-default|grammar-[a-z0-9-]+)-/;
const behaviorFilename = /^(toolcall-quality|agentic|frameworks|opencode|pi|openclaw|hermes)-default-/;

function bucketFor(filename) {
  if (filename.includes('adaptive-xml')) {
    return buckets.forcedParserCompatibility;
  }
  if (behaviorFilename.test(filename)) {
    return buckets.modelAgentBehaviorQuality;
  }
  if (nativeFilename.test(filename)) {
    return buckets.nativeProtocolConformance;
  }
  throw new Error(`${filename} has no explicit Promptfoo result category`);
}

function normalizedFailureCause(value) {
  if (typeof value !== 'string') {
    return null;
  }
  const normalized = value.trim().toLowerCase().replaceAll('_', ' ');
  return failureCauseLabels.includes(normalized) ? normalized : null;
}

function explicitFailureCause(result) {
  const metadataSources = [result?.testCase?.metadata, result?.metadata, result?.response?.metadata];
  for (const metadata of metadataSources) {
    for (const key of ['failureCause', 'failure_cause', 'failureClassification']) {
      const cause = normalizedFailureCause(metadata?.[key]);
      if (cause) {
        return cause;
      }
    }
  }
  return null;
}

function failureText(result) {
  return [
    result?.error,
    result?.gradingResult?.reason,
    result?.response?.error,
    ...(result?.gradingResult?.componentResults || []).map((component) => component?.reason),
  ].filter((value) => typeof value === 'string').join('\n');
}

function failureCauseFor(filename, bucket, result) {
  const explicit = explicitFailureCause(result);
  if (explicit) {
    return explicit;
  }

  // A forced parser is an experiment, not evidence against the native model
  // path. Only explicit per-case evidence above may override this attribution.
  if (bucket === buckets.forcedParserCompatibility) {
    return 'forced-parser experiment';
  }

  const evidence = failureText(result);
  if (/\b(test harness|fixture|provider configuration|missing required environment|cannot find module)\b/i.test(evidence) ||
      /\b(?:promptfoo|provider) (?:configuration|error|failed|missing|invalid)\b/i.test(evidence) ||
      /\b(custom assertion|javascript) (?:threw|syntax error)\b/i.test(evidence)) {
    return 'test harness';
  }
  if (/\b(?:ECONNREFUSED|ETIMEDOUT|socket hang up|server exited|server crash|internal server error|HTTP 5\d\d|malformed (?:JSON|SSE)|invalid (?:JSON|SSE) response|transport error)\b/i.test(evidence)) {
    return 'engine/runtime likely';
  }
  if (bucket === buckets.modelAgentBehaviorQuality) {
    return 'model behavior likely';
  }
  if (/^(structured|structured-stress|grammar-[a-z0-9-]+)-/.test(filename)) {
    return 'engine/runtime likely';
  }
  return 'unresolved';
}

function caseName(result) {
  return result?.testCase?.description || result?.prompt?.raw || 'unnamed case';
}

function recordFailureCause(cause, filename, result) {
  const name = caseName(result);
  const taxonomyBucket = failureTaxonomy[cause];
  taxonomyBucket.count += 1;
  taxonomyBucket.uniqueFailingCases.add(name);
  taxonomyBucket.records.push({
    file: filename,
    case: name,
    reason: result?.error || result?.gradingResult?.reason || result?.response?.error || null,
  });
}

const filenames = fs.readdirSync(outDir)
  .filter((filename) => filename.endsWith(reportSuffix))
  .filter((filename) => !filename.startsWith('promptfoo-summary-'))
  .filter((filename) => !filename.includes('.classified.'))
  .filter((filename) => minimumMtimeMs === null ||
    fs.statSync(path.join(outDir, filename)).mtimeMs >= minimumMtimeMs)
  .sort();

const skipPath = path.join(outDir, 'capability-skips.json');
let capabilitySkips = [];
if (fs.existsSync(skipPath) &&
    (minimumMtimeMs === null || fs.statSync(skipPath).mtimeMs >= minimumMtimeMs)) {
  capabilitySkips = JSON.parse(fs.readFileSync(skipPath, 'utf8'));
  if (!Array.isArray(capabilitySkips) || capabilitySkips.some((record) =>
    record.status !== 'SKIP' || record.evidence?.status !== 'known' ||
    record.evidence?.requested_model !== model || !record.scope ||
    !Array.isArray(record.missing_capabilities) || record.missing_capabilities.length === 0)) {
    throw new Error('invalid current-run capability skip inventory');
  }
}
if (filenames.length === 0 && capabilitySkips.length === 0) {
  throw new Error('no current-run Promptfoo JSON reports found');
}

for (const filename of filenames) {
  const document = JSON.parse(fs.readFileSync(path.join(outDir, filename), 'utf8'));
  const stats = document?.results?.stats;
  const results = document?.results?.results;
  if (!stats || !Array.isArray(results)) {
    throw new Error(`${filename} is not a Promptfoo JSON report`);
  }

  const successes = Number(stats.successes || 0);
  const failures = Number(stats.failures || 0);
  const errors = Number(stats.errors || 0);
  for (const [name, value] of Object.entries({ successes, failures, errors })) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new Error(`${filename} has invalid ${name} count: ${value}`);
    }
  }
  const cases = successes + failures + errors;
  const failedResults = results.filter((result) => result?.success === false);
  if (results.length !== cases || failedResults.length !== failures + errors) {
    throw new Error(
      `${filename} stats/results mismatch: stats=${cases}, results=${results.length}, ` +
      `failed stats=${failures + errors}, failed results=${failedResults.length}`
    );
  }

  const bucket = bucketFor(filename);
  bucket.files.push(filename);
  bucket.successes += successes;
  bucket.failures += failures;
  bucket.errors += errors;
  for (const result of failedResults) {
    bucket.uniqueFailingCases.add(caseName(result));
    recordFailureCause(failureCauseFor(filename, bucket, result), filename, result);
  }
}

function serializableBucket(bucket) {
  const cases = bucket.successes + bucket.failures + bucket.errors;
  return {
    title: bucket.title,
    description: bucket.description,
    files: bucket.files,
    fileCount: bucket.files.length,
    cases,
    successes: bucket.successes,
    failures: bucket.failures,
    errors: bucket.errors,
    uniqueFailingCases: bucket.uniqueFailingCases.size,
    passRate: cases === 0 ? null : Number((bucket.successes / cases).toFixed(4)),
  };
}

const summary = {
  capabilitySkips,
  schemaVersion: 1,
  model,
  generatedAt: new Date().toISOString(),
  note: 'The three categories are intentionally not combined into a single quality score.',
  categories: {
    nativeProtocolConformance: serializableBucket(buckets.nativeProtocolConformance),
    modelAgentBehaviorQuality: serializableBucket(buckets.modelAgentBehaviorQuality),
    forcedParserCompatibility: serializableBucket(buckets.forcedParserCompatibility),
  },
  failureTaxonomy: {
    note: 'Broad, ownership-agnostic attribution only; engine/runtime likely does not decide whether the fix belongs in AFM, AFMKit, MLXSwift, or another runtime dependency.',
    totalFailuresAndErrors: Object.values(failureTaxonomy)
      .reduce((total, bucket) => total + bucket.count, 0),
    buckets: Object.fromEntries(Object.entries(failureTaxonomy).map(([label, bucket]) => [label, {
      count: bucket.count,
      uniqueFailingCases: bucket.uniqueFailingCases.size,
      records: bucket.records,
    }])),
  },
};

const categoryFailureTotal = Object.values(summary.categories)
  .reduce((total, category) => total + category.failures + category.errors, 0);
if (summary.failureTaxonomy.totalFailuresAndErrors !== categoryFailureTotal) {
  throw new Error(
    `failure taxonomy mismatch: taxonomy=${summary.failureTaxonomy.totalFailuresAndErrors}, ` +
    `categories=${categoryFailureTotal}`
  );
}

fs.writeFileSync(`${outputBase}.json`, `${JSON.stringify(summary, null, 2)}\n`);

const rows = Object.values(summary.categories).map((category) => {
  const rate = category.passRate === null ? 'n/a' : `${(category.passRate * 100).toFixed(1)}%`;
  return `| ${category.title} | ${category.cases} | ${category.successes} | ${category.failures} | ${category.errors} | ${category.uniqueFailingCases} | ${rate} |`;
});
const markdown = `# Promptfoo result categories\n\n` +
  `- Model: \`${model}\`\n` +
  `- Generated: ${summary.generatedAt}\n\n` +
  `- Capability-skipped suite/profile scopes: ${capabilitySkips.length} (not executed; not conformance passes; excluded from case totals)\n\n` +
  `These categories are intentionally reported separately. Forced-parser experiments are not part of native protocol conformance, and behavioral preferences are not OpenAI API invariants.\n\n` +
  `| Category | Cases | Pass | Fail | Error | Unique failing cases | Pass rate |\n` +
  `|---|---:|---:|---:|---:|---:|---:|\n` +
  `${rows.join('\n')}\n\n` +
  `## Broad failure taxonomy\n\n` +
  `These are broad, ownership-agnostic attributions. Engine/runtime likely identifies the failing layer without deciding whether the fix belongs in AFM, AFMKit, MLXSwift, or another runtime dependency.\n\n` +
  `| Attribution | Failures and errors | Unique failing cases |\n` +
  `|---|---:|---:|\n` +
  `${Object.entries(summary.failureTaxonomy.buckets).map(([label, bucket]) =>
    `| ${label} | ${bucket.count} | ${bucket.uniqueFailingCases} |`
  ).join('\n')}\n\n` +
  Object.values(summary.categories).map((category) =>
    `## ${category.title}\n\n${category.description}\n\nFiles: ${category.fileCount}\n`
  ).join('\n');
fs.writeFileSync(`${outputBase}.md`, markdown);

console.log(`Promptfoo categorized summary: ${outputBase}.md`);
