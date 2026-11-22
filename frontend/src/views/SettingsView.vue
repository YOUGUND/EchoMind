<script setup>
import { computed, ref } from 'vue'
import { KeyRound, Server, ShieldCheck, RefreshCw, CheckCircle2, LogIn, LogOut, UserPlus } from 'lucide-vue-next'
import { api, apiBase, token, setToken, jwt, setJwt, setAuthUser, clearAuth, authUser } from '../lib/api'

const props = defineProps({ workspace: Object })
const { health, refresh, notify, syncAuth, hideLocalSessions } = props.workspace

const accessToken = ref(token())
const saving = ref(false)
const authMode = ref('login')
const email = ref('')
const password = ref('')
const authBusy = ref(false)
const authError = ref('')
const loggedIn = ref(authUser())

const isMulti = computed(() => health.value?.storage_mode === 'multi' || health.value?.auth === 'jwt')

async function save() {
  saving.value = true
  setToken(accessToken.value)
  await refresh()
  saving.value = false
  if (!props.workspace.connectionError.value) notify('访问设置已保存，后端连接正常')
}

async function submitAuth() {
  authBusy.value = true
  authError.value = ''
  try {
    const path = authMode.value === 'register' ? '/auth/register' : '/auth/login'
    const data = await api(path, { method: 'POST', body: { email: email.value.trim(), password: password.value } })
    setJwt(data.access_token)
    setAuthUser({ user_id: data.user_id, email: data.email })
    loggedIn.value = { user_id: data.user_id, email: data.email }
    syncAuth?.()
    await refresh()
    notify(authMode.value === 'register' ? '注册成功，已登录' : '登录成功')
  } catch (error) {
    authError.value = error.message
  } finally {
    authBusy.value = false
  }
}

function logout() {
  clearAuth()
  loggedIn.value = null
  // 只隐藏界面；同一账号再登录会从本机账号桶恢复会话
  hideLocalSessions?.({ keepBlank: true })
  syncAuth?.()
  notify('已退出登录')
}
</script>

<template>
  <div class="standard-page settings-page">
    <div class="page-heading">
      <div>
        <div class="eyebrow">WORKSPACE SETTINGS</div>
        <h1>工作台设置<span class="heading-dot">.</span></h1>
        <p>连接你的工作空间，了解当前运行配置。</p>
      </div>
    </div>

    <div class="settings-grid">
      <section v-if="isMulti" class="surface">
        <div class="section-heading">
          <h2>多用户登录</h2>
          <LogIn :size="20" class="muted" />
        </div>
        <template v-if="loggedIn">
          <div class="public-access-card">
            <ShieldCheck :size="22" />
            <div>
              <strong>已登录</strong>
              <p>{{ loggedIn.email }} · user_id={{ loggedIn.user_id.slice(0, 8) }}…</p>
            </div>
          </div>
          <p class="small muted">会话与知识库按 JWT 内的 user_id 隔离；客户端伪造 user_id 无效。</p>
          <button class="secondary" @click="logout"><LogOut :size="16" /> 退出登录</button>
        </template>
        <template v-else>
          <p class="muted">STORAGE_MODE=multi：请注册或登录后，对话与知识库才会按用户隔离。</p>
          <div class="auth-tabs">
            <button type="button" class="quiet" :class="{ active: authMode === 'login' }" @click="authMode = 'login'">登录</button>
            <button type="button" class="quiet" :class="{ active: authMode === 'register' }" @click="authMode = 'register'">注册</button>
          </div>
          <form @submit.prevent="submitAuth">
            <label class="field">邮箱<input v-model="email" type="email" autocomplete="username" placeholder="you@example.com" required /></label>
            <label class="field">密码<input v-model="password" type="password" autocomplete="current-password" placeholder="至少 8 位" minlength="8" required /></label>
            <p v-if="authError" class="small warning-text">{{ authError }}</p>
            <button class="primary" :disabled="authBusy">
              <component :is="authMode === 'register' ? UserPlus : LogIn" :size="16" />
              {{ authBusy ? '提交中…' : (authMode === 'register' ? '注册并登录' : '登录') }}
            </button>
          </form>
        </template>
      </section>

      <section class="surface">
        <div class="section-heading">
          <h2>工作空间访问</h2>
          <KeyRound :size="20" class="muted" />
        </div>
        <template v-if="health?.public_demo || isMulti">
          <div class="public-access-card">
            <ShieldCheck :size="22" />
            <div>
              <strong>{{ isMulti ? '多用户 JWT 模式' : '公开演示已开启' }}</strong>
              <p v-if="isMulti">身份由登录 Token 认定；工作台访问码在此模式下可不填。</p>
              <p v-else>打开网页即可体验，无需填写访问码。模型密钥仍只保存在后端，访客无法查看。</p>
            </div>
          </div>
          <p v-if="!isMulti" class="small muted">模型接口继续受全局频率与并发限制。公开访客共享临时知识空间，服务休眠或重启后数据清除。</p>
        </template>
        <template v-else>
          <p class="muted">此工作台使用访问码保护，避免其他人消耗你的模型额度。</p>
          <form @submit.prevent="save">
            <label class="field">工作台访问码<input type="password" v-model="accessToken" autocomplete="off" placeholder="填写 APP_ACCESS_TOKEN 对应的访问码" /></label>
            <p class="small muted">这是工作台访问码，不是模型 API 密钥。仅保存在当前浏览器会话。</p>
            <button class="primary" :disabled="saving"><ShieldCheck :size="16" />{{ saving ? '正在验证连接…' : '保存并连接' }}</button>
          </form>
        </template>
      </section>

      <section class="surface">
        <div class="section-heading">
          <h2>运行配置</h2>
          <Server :size="20" class="muted" />
        </div>
        <dl class="config-list">
          <div><dt>后端状态</dt><dd><span :class="['tag', health ? '' : 'warning']">{{ health ? '服务在线' : '未连接' }}</span></dd></div>
          <div><dt>访问方式</dt><dd>{{ isMulti ? 'JWT 多用户' : (health?.public_demo ? '公开演示 · 无访问码' : '访问码保护') }}</dd></div>
          <div><dt>存储模式</dt><dd>{{ health?.storage_mode || 'memory' }}</dd></div>
          <div><dt>知识检索</dt><dd>{{ health?.retrieval || '—' }}</dd></div>
          <div><dt>模型</dt><dd>{{ health?.model || '连接后查看' }}</dd></div>
          <div><dt>模型密钥</dt><dd>{{ health?.model_configured ? '后端已配置 · 不向浏览器公开' : '尚未配置' }}</dd></div>
          <div><dt>API 地址</dt><dd class="mono">{{ apiBase }}</dd></div>
          <div><dt>JWT</dt><dd>{{ jwt() ? '已保存于浏览器会话' : '未登录' }}</dd></div>
        </dl>
        <button class="quiet" @click="refresh"><RefreshCw :size="15" /> 检查连接</button>
      </section>
    </div>

    <section class="surface about-project">
      <div class="eyebrow">ABOUT ECHOMIND</div>
      <h2>EchoMind，把 Agent 的过程变得可理解。</h2>
      <p>这是基于 LangGraph 的多 Agent 客服编排工作台：意图融合、主辅路由、工具白名单、上下文预算与评测门禁串成一条可观察链路。前端用 Vue 呈现执行过程，后端用 FastAPI + PostgreSQL + Milvus 承接多用户数据。</p>
      <div class="about-points">
        <span><CheckCircle2 :size="16" /> LangGraph 编排</span>
        <span><CheckCircle2 :size="16" /> JWT 用户隔离</span>
        <span><CheckCircle2 :size="16" /> 结构化执行记录</span>
      </div>
      <div class="subtle-note">本地请先启动 docker-compose.multi.yml（Postgres / Milvus），再配置 backend/.env.local。模型密钥只保存在服务端；公开演示环境的数据边界以部署说明为准。</div>
    </section>
  </div>
</template>

<style scoped>
.auth-tabs { display: flex; gap: 8px; margin: 12px 0; }
.auth-tabs .quiet.active { opacity: 1; font-weight: 600; text-decoration: underline; }
.warning-text { color: #b45309; margin: 8px 0; }
.field { display: grid; gap: 6px; margin-bottom: 12px; }
</style>
