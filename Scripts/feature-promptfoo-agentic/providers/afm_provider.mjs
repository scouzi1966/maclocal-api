import { spawn } from 'node:child_process';

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

function normalizeToolCalls(toolCalls) {
  return (toolCalls || []).map((toolCall) => {
    const functionPayload = toolCall?.function || {};
    const rawArguments = functionPayload.arguments;
    const parsedArguments =
      typeof rawArguments === 'string' ? (safeJsonParse(rawArguments) ?? rawArguments) : rawArguments;

    return {
      ...(toolCall || {}),
      function: {
        ...functionPayload,
        arguments: parsedArguments,
      },
    };
  });
}

function resolveBaseUrl(config) {
  if (config.baseUrl) return config.baseUrl;
  if (config.baseUrlEnv) {
    const value = readEnv(config.baseUrlEnv);
    if (!value) {
      throw new Error(`Missing required environment variable ${config.baseUrlEnv}`);
    }
    return value;
  }
  throw new Error('AFM provider requires config.baseUrl or config.baseUrlEnv');
}

function resolveModel(config, vars) {
  return vars.model || config.model || readEnv(config.modelEnv || 'AFM_MODEL');
}

function extractOutput(config, responseBody) {
  const choice = responseBody?.choices?.[0] || {};
  const message = choice.message || {};
  const mode = config.extract || 'content';
  const normalizedToolCalls = normalizeToolCalls(message.tool_calls);

  if (mode === 'tool_calls') {
    return JSON.stringify(normalizedToolCalls);
  }

  if (mode === 'normalized_message') {
    return JSON.stringify({
      role: message.role || 'assistant',
      content: message.content || '',
      tool_calls: normalizedToolCalls,
      finish_reason: choice.finish_reason || null,
      refusal: message.refusal || null,
    });
  }

  if (mode === 'full_response') {
    return JSON.stringify(responseBody);
  }

  return message.content || '';
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
    throw new Error(`AFM returned non-JSON response: ${text}`);
  }
  return parsed;
}

async function runCli(binary, args, env) {
  return await new Promise((resolve, reject) => {
    const child = spawn(binary, args, {
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(`CLI exited ${code}: ${stderr || stdout}`));
      }
    });
  });
}

export default class AfmPromptfooProvider {
  constructor(options = {}) {
    this.options = options;
    this.config = options.config || {};
    this.label = options.label;
  }

  id() {
    return this.options.id || 'file://Scripts/feature-promptfoo-agentic/providers/afm_provider.mjs';
  }

  async callApi(prompt, context) {
    const vars = context?.vars || {};
    const transport = this.config.transport || 'api';
    const started = Date.now();

    if (transport === 'cli-guided-json') {
      const binary = this.config.binary || readEnv(this.config.binaryEnv || 'AFM_BINARY', '.build/arm64-apple-macosx/release/afm');
      const model = resolveModel(this.config, vars);
      const schema = vars.guided_json || vars.schema;
      if (!model) {
        throw new Error('Missing AFM model for CLI guided-json provider');
      }
      if (!schema) {
        throw new Error('CLI guided-json provider requires vars.guided_json or vars.schema');
      }

      const args = [
        'mlx',
        '-m',
        model,
        '-s',
        prompt,
        '--guided-json',
        JSON.stringify(schema),
        '--max-tokens',
        String(vars.max_tokens || this.config.maxTokens || 400),
        '--temperature',
        String(vars.temperature ?? this.config.temperature ?? 0),
      ];

      if (this.config.noThink || readEnv('AFM_NO_THINK') === '1') {
        args.push('--no-think');
      }

      const env = { ...process.env };
      if (this.config.modelCacheEnv) {
        const cachePath = readEnv(this.config.modelCacheEnv);
        if (cachePath) {
          env.MACAFM_MLX_MODEL_CACHE = cachePath;
        }
      }

      const { stdout, stderr } = await runCli(binary, args, env);
      return {
        output: stdout.trim(),
        latencyMs: Date.now() - started,
        metadata: {
          stderr,
          transport,
          binary,
          args,
        },
      };
    }

    const baseUrl = resolveBaseUrl(this.config).replace(/\/+$/, '');
    const model = resolveModel(this.config, vars);
    if (!model) {
      throw new Error('Missing AFM model for API provider');
    }

    const body = {
      model,
      messages: vars.messages || [
        ...(vars.system_prompt ? [{ role: 'system', content: vars.system_prompt }] : []),
        { role: 'user', content: prompt },
      ],
      max_tokens: vars.max_tokens || this.config.maxTokens || 400,
      temperature: vars.temperature ?? this.config.temperature ?? 0,
      stream: false,
    };

    if (vars.response_format) body.response_format = vars.response_format;
    if (vars.tools) body.tools = vars.tools;
    if (vars.tool_choice !== undefined) body.tool_choice = vars.tool_choice;
    if (vars.stop) body.stop = vars.stop;
    if (vars.seed !== undefined) body.seed = vars.seed;

    const responseBody = await postJson(`${baseUrl}/chat/completions`, body);

    return {
      output: extractOutput(this.config, responseBody),
      latencyMs: Date.now() - started,
      metadata: {
        transport,
        baseUrl,
        requestBody: body,
        responseBody,
        usage: responseBody.usage || null,
      },
    };
  }
}
