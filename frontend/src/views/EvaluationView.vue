<script setup>
import { onMounted, ref, computed, watch } from 'vue'
import { FlaskConical, Play, Download, CheckCircle2, XCircle, LoaderCircle, RefreshCw, Layers3 } from 'lucide-vue-next'
import { api, downloadJson } from '../lib/api'

const props = defineProps({ workspace: Object })

const tab = ref('platform') // platform | lab
const labMode = ref('intent')
const question = ref('购买三天后想退款，应该如何申请？')
const intent = ref('refund')
const intents = [
  'refund', 'invoice', 'order_status', 'logistics', 'technical_login',
  'technical_crash', 'payment_issue', 'greeting', 'query', 'other',
  'complaint', 'request', 'account', 'human_handoff', 'escalation',
]

const suites = ref([])
const suiteId = ref('intent_core')
const failUnder = ref(0.75)
const updateBaseline = ref(false)
const runs = ref([])
const busy = ref(false)
const report = ref(null)
const error = ref('')
const selectedRunId = ref('')

const labels = {
  relevance: '相关性',
  accuracy: '准确性',
  completeness: '完整性',
  helpfulness: '有用性',
  intent_accuracy: '意图准确率',
}

const selectedSuite = computed(() => suites.value.find((item) => item.id === suiteId.value) || null)

async function loadSuites() {
  const data = await api('/eval/suites')
  suites.value = data.items || []
  if (data.default_suite && suites.value.find((item) => item.id === data.default_suite)) {
    suiteId.value = data.default_suite
  } else if (!suites.value.find((item) => item.id === suiteId.value) && suites.value[0]) {
    suiteId.value = suites.value[0].id
  }
  const current = suites.value.find((item) => item.id === suiteId.value)
  if (current && current.fail_under != null) {
    failUnder.value = current.fail_under
  }
}

async function loadRuns() {
  const data = await api('/eval/runs?limit=12')
  runs.value = data.items || []
}

async function refreshPlatform() {
  error.value = ''
  try {
    await Promise.all([loadSuites(), loadRuns()])
  } catch (e) {
    error.value = e.message
  }
}

async function runLab() {
  busy.value = true
  error.value = ''
  report.value = null
  try {
    const body = labMode.value === 'intent'
      ? { intent_cases: [{ message: question.value, expected_intent: intent.value }], dialog_cases: [], save_run: true }
      : { intent_cases: [], dialog_cases: [{ question: question.value }], save_run: true }
    report.value = await api('/eval/run', { method: 'POST', body })
    props.workspace.notify(report.value.gate_failed ? '单案例评测完成（未达门禁）' : '单案例评测完成')
    await loadRuns()
    await props.workspace.refresh()
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
  }
}

async function runSuite() {
  if (!suiteId.value) return
  busy.value = true
  error.value = ''
  report.value = null
  try {
    report.value = await api('/eval/run', {
      method: 'POST',
      body: {
        suite_id: suiteId.value,
        fail_under: Number(failUnder.value),
        save_run: true,
        update_baseline: updateBaseline.value,
        collect_regressions: true,
      },
    })
    selectedRunId.value = report.value.run_id || ''
    props.workspace.notify(
      report.value.gate_failed
        ? `套件 ${suiteId.value} 未过门禁`
        : `套件 ${suiteId.value} 评测完成`,
    )
    await loadRuns()
    await props.workspace.refresh()
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
  }
}

async function openRun(runId) {
  if (!runId) return
  busy.value = true
  error.value = ''
  try {
    const detail = await api(`/eval/runs/${runId}`)
    selectedRunId.value = runId
    report.value = {
      ...(detail.report || {}),
      run_id: detail.run_id,
      suite_id: detail.suite_id,
      suite_version: detail.suite_version,
      gate: detail.gate,
      gate_failed: detail.gate?.enabled && !detail.gate?.passed,
    }
  } catch (e) {
    error.value = e.message
  } finally {
    busy.value = false
  }
}

watch(suiteId, (id) => {
  const current = suites.value.find((item) => item.id === id)
  if (current && current.fail_under != null) {
    failUnder.value = current.fail_under
  }
})

onMounted(refreshPlatform)
</script>

<template>
  <div class="standard-page">
    <div class="page-heading">
      <div>
        <div class="eyebrow">EVAL PLATFORM</div>
        <h1>评测中台<span class="heading-dot">.</span></h1>
        <p>套件跑批、报告落盘、门禁与坏案例回流——和 Agent 主链路共用同一套评测引擎。</p>
      </div>
      <div class="heading-actions">
        <button class="secondary" :disabled="busy" @click="refreshPlatform">
          <RefreshCw :size="15" /> 刷新
        </button>
        <span class="tag neutral">Suite · Runs · Gate</span>
      </div>
    </div>

    <div class="segmented" style="margin-bottom: 18px; max-width: 360px">
      <button :class="{ active: tab === 'platform' }" @click="tab = 'platform'">套件跑批</button>
      <button :class="{ active: tab === 'lab' }" @click="tab = 'lab'">单案例实验</button>
    </div>

    <div class="evaluation-grid">
      <section class="surface">
        <template v-if="tab === 'platform'">
          <div class="section-heading">
            <h2>选择套件并跑批</h2>
            <Layers3 :size="20" class="muted" />
          </div>
          <form @submit.prevent="runSuite">
            <label class="field">
              评测套件
              <select v-model="suiteId">
                <option v-for="item in suites" :key="item.id" :value="item.id">
                  {{ item.name }} ({{ item.id }})
                </option>
              </select>
            </label>
            <p v-if="selectedSuite" class="subtle-note">
              {{ selectedSuite.purpose || selectedSuite.description || '无描述' }}
              · 意图 {{ selectedSuite.intent_cases }} · 对话 {{ selectedSuite.dialog_cases }}
              · v{{ selectedSuite.version }}
              <span v-if="selectedSuite.tags?.length"> · tags: {{ selectedSuite.tags.join(', ') }}</span>
            </p>
            <label class="field">
              门禁阈值 fail_under（通过率）
              <input v-model.number="failUnder" type="number" min="0" max="1" step="0.05" />
            </label>
            <label class="field" style="display: flex; align-items: center; gap: 8px">
              <input v-model="updateBaseline" type="checkbox" style="width: auto" />
              本次通过后写入 baseline
            </label>
            <div class="subtle-note">
              套件来自 <code>data/eval/suites</code>；跑次写入 <code>data/eval/runs</code>；失败 case 回流到 regressions。
              此操作会调用模型。
            </div>
            <button class="primary full-width" :disabled="busy || !suiteId">
              <LoaderCircle v-if="busy" :size="17" class="spin" />
              <Play v-else :size="16" />
              {{ busy ? '套件评测进行中…' : '运行套件' }}
            </button>
          </form>

          <div class="section-heading" style="margin-top: 28px">
            <h2>历史跑次</h2>
          </div>
          <div v-if="!runs.length" class="subtle-note">暂无跑次。先运行一个套件。</div>
          <button
            v-for="item in runs"
            :key="item.run_id"
            class="history-item"
            style="width: 100%; margin-bottom: 8px; text-align: left"
            :class="{ selected: selectedRunId === item.run_id }"
            @click="openRun(item.run_id)"
          >
            <span>
              <strong>{{ item.suite_id || 'adhoc' }}</strong>
              <small style="display: block" class="muted">
                {{ item.run_id }} · 通过率 {{ item.pass_rate == null ? '—' : Math.round(item.pass_rate * 100) + '%' }}
              </small>
            </span>
          </button>
        </template>

        <template v-else>
          <div class="section-heading">
            <h2>设计一项实验</h2>
            <FlaskConical :size="20" class="muted" />
          </div>
          <div class="segmented">
            <button :class="{ active: labMode === 'intent' }" @click="labMode = 'intent'">意图识别</button>
            <button :class="{ active: labMode === 'quality' }" @click="labMode = 'quality'">回复质量</button>
          </div>
          <form @submit.prevent="runLab">
            <label class="field">
              测试问题
              <textarea v-model="question" rows="5" required maxlength="8000" />
            </label>
            <label v-if="labMode === 'intent'" class="field">
              预期意图
              <select v-model="intent">
                <option v-for="item in intents" :key="item" :value="item">{{ item }}</option>
              </select>
            </label>
            <div class="subtle-note">
              {{ labMode === 'intent' ? '检查预测意图与预期标签是否一致。' : 'Agent 生成回复后，由模型评判相关性、准确性、完整性与有用性。' }}
              单案例也会落盘到 runs，便于和学习套件对比。
            </div>
            <button class="primary full-width" :disabled="busy || !question.trim()">
              <LoaderCircle v-if="busy" :size="17" class="spin" />
              <Play v-else :size="16" />
              {{ busy ? '正在运行评测…' : '运行这项实验' }}
            </button>
          </form>
        </template>

        <div v-if="error" class="error-box" role="alert">{{ error }}</div>
      </section>

      <section class="surface report-surface">
        <div class="section-heading">
          <h2>评测报告</h2>
          <button class="quiet" :disabled="!report" @click="downloadJson('echomind-evaluation.json', report)">
            <Download :size="15" /> 导出
          </button>
        </div>

        <div v-if="!report" class="retrieval-empty">
          <FlaskConical :size="38" :stroke-width="1.2" />
          <h3>{{ busy ? '评测正在进行' : '选择套件或单案例开始' }}</h3>
          <p>结果来自实际执行，并写入评测中台 runs。</p>
        </div>

        <template v-else>
          <div v-if="report.gate_failed || (report.gate && report.gate.enabled && !report.gate.passed)" class="error-box" role="status">
            门禁未通过：{{ report.gate?.reason || `pass_rate 低于阈值 ${report.gate?.fail_under}` }}
          </div>
          <div class="report-summary">
            <strong>{{ Math.round((report.pass_rate || 0) * 100) }}<small>%</small></strong>
            <div>
              通过率
              <p>
                {{ report.passed }} / {{ report.total }} 项通过
                <span v-if="report.run_id"> · {{ report.run_id }}</span>
                <span v-if="report.suite_id"> · {{ report.suite_id }}</span>
              </p>
            </div>
          </div>

          <div class="score-grid">
            <div v-for="(score, key) in report.avg_scores || {}" :key="key">
              <span>{{ labels[key] || key }}</span>
              <strong>{{ (score * 100).toFixed(1) }}%</strong>
              <div class="score-track"><i :style="{ width: `${Math.min(100, score * 100)}%` }" /></div>
            </div>
          </div>

          <article v-for="result in report.results || []" :key="result.test_id" class="evaluation-result">
            <component
              :is="result.passed ? CheckCircle2 : XCircle"
              :size="19"
              :class="result.passed ? 'success-text' : 'danger'"
            />
            <div>
              <strong>{{ result.test_id }}</strong>
              <p>{{ result.detail }}</p>
              <details v-if="result.metadata">
                <summary>查看案例与判定依据</summary>
                <pre>{{ JSON.stringify(result.metadata, null, 2) }}</pre>
              </details>
            </div>
          </article>

          <div v-if="report.regressions?.length" class="subtle-note">
            <strong>回归告警</strong>
            <p v-for="item in report.regressions" :key="item">{{ item }}</p>
          </div>
          <div v-if="report.recommendations?.length" class="subtle-note">
            <strong>改进建议</strong>
            <p v-for="item in report.recommendations" :key="item">{{ item }}</p>
          </div>
          <div v-if="report.regressions_added" class="subtle-note">
            已回流 {{ report.regressions_added }} 条坏案例到 regressions 套件。
          </div>
        </template>
      </section>
    </div>
  </div>
</template>
