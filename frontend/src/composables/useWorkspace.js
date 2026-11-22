import { ref, computed, watch, onMounted } from 'vue'
import { api, authUser, downloadJson } from '../lib/api'

const LEGACY_STORAGE_KEY = 'epochflow.sessions.v1'
const GUEST_STORAGE_KEY = 'echomind.sessions.v1.guest'

function sessionKeyFor(user) {
  if (user?.user_id) return `echomind.sessions.v1.user.${user.user_id}`
  return GUEST_STORAGE_KEY
}

function restore(key) {
  try {
    return JSON.parse(sessionStorage.getItem(key) || '[]').slice(0, 30)
  } catch {
    return []
  }
}

export function useWorkspace() {
  const view = ref('chat')
  const currentUser = ref(authUser())
  const storageKey = ref(sessionKeyFor(currentUser.value))
  // multi 未登录：界面不展示任何历史，账号桶仍保留在 sessionStorage
  const conversations = ref(
    currentUser.value ? restore(storageKey.value) : []
  )
  const activeId = ref(conversations.value[0]?.id || '')
  const active = computed(() => conversations.value.find(item => item.id === activeId.value))
  const health = ref(null), monitor = ref(null), skills = ref(null), notice = ref(''), busy = ref(false), connectionError = ref('')
  const catalog = ref(null), selectedModel = ref(sessionStorage.getItem('echomind.agentModel') || sessionStorage.getItem('epochflow.agentModel') || '')
  watch(selectedModel, value => sessionStorage.setItem('echomind.agentModel', value))
  const agentModels = computed(() => (catalog.value?.items || []).filter(item => item.agent))
  async function loadModels() { catalog.value = await api('/models'); return catalog.value }
  function useModel(id) { selectedModel.value = id; createConversation(); notify('新会话已选择 ' + id.split('/').at(-1)) }
  let noticeTimer
  // Local UUID is display-only in multi mode; server trusts JWT subject.
  const userId = sessionStorage.getItem('echomind.user') || sessionStorage.getItem('epochflow.user') || crypto.randomUUID()
  sessionStorage.setItem('echomind.user', userId)

  let persistSessions = Boolean(currentUser.value)

  watch(conversations, value => {
    if (!persistSessions) return
    try {
      sessionStorage.setItem(storageKey.value, JSON.stringify(value.slice(0, 30)))
    } catch { /* Quota must not prevent chat. */ }
  }, { deep: true })

  function notify(message) { notice.value = message; clearTimeout(noticeTimer); noticeTimer = setTimeout(() => { notice.value = '' }, 5500) }

  function createConversation() {
    const conversation = { id: crypto.randomUUID(), title: '新的 Agent 实验', serverId: '', messages: [], createdAt: new Date().toISOString() }
    conversations.value.unshift(conversation); conversations.value = conversations.value.slice(0, 30); activeId.value = conversation.id; view.value = 'chat'
    return conversation
  }

  function selectConversation(id) { activeId.value = id; view.value = 'chat' }

  function deleteConversation() {
    if (busy.value) return
    conversations.value = conversations.value.filter(item => item.id !== activeId.value)
    activeId.value = conversations.value[0]?.id || ''; if (!activeId.value) createConversation()
    notify('已移除此浏览器中的会话记录')
  }

  /** 仅清空界面；不删除各账号在本机保存的会话桶。 */
  function hideLocalSessions({ keepBlank = true } = {}) {
    persistSessions = false
    storageKey.value = GUEST_STORAGE_KEY
    conversations.value = []
    activeId.value = ''
    if (keepBlank) createConversation()
  }

  /** 真正删除本机会话（设置里若需要可再暴露）；默认退出登录不要调用。 */
  function clearLocalSessions({ keepBlank = true, wipeAllUsers = false } = {}) {
    try {
      sessionStorage.removeItem(LEGACY_STORAGE_KEY)
      sessionStorage.removeItem(GUEST_STORAGE_KEY)
      if (wipeAllUsers) {
        for (let i = sessionStorage.length - 1; i >= 0; i -= 1) {
          const key = sessionStorage.key(i)
          if (key && (key.startsWith('echomind.sessions.v1') || key.startsWith('epochflow.sessions.v1'))) sessionStorage.removeItem(key)
        }
      } else if (currentUser.value?.user_id) {
        sessionStorage.removeItem(sessionKeyFor(currentUser.value))
      }
    } catch { /* ignore */ }
    hideLocalSessions({ keepBlank })
  }

  function loadSessionsForUser(user) {
    storageKey.value = sessionKeyFor(user)
    let items = user ? restore(storageKey.value) : []
    // 兼容：旧版本只写在共用 key 上时，首次登录迁到账号桶
    if (user && !items.length) {
      const legacy = restore(LEGACY_STORAGE_KEY)
      if (legacy.length) {
        items = legacy
        try {
          sessionStorage.setItem(storageKey.value, JSON.stringify(legacy.slice(0, 30)))
          sessionStorage.removeItem(LEGACY_STORAGE_KEY)
        } catch { /* ignore */ }
      }
    }
    persistSessions = Boolean(user)
    conversations.value = items
    activeId.value = items[0]?.id || ''
    if (!activeId.value) createConversation()
  }

  async function refresh() {
    connectionError.value = ''
    try {
      health.value = await api('/health')
      const results = await Promise.allSettled([api('/monitor'), api('/skills'), loadModels()])
      if (results[0].status === 'fulfilled') monitor.value = results[0].value
      else connectionError.value = results[0].reason.message
      if (results[1].status === 'fulfilled') skills.value = results[1].value
      if (health.value?.storage_mode === 'multi' && !currentUser.value) {
        hideLocalSessions({ keepBlank: true })
      }
    } catch (error) { health.value = null; connectionError.value = error.message }
  }

  async function send(text, retry = false) {
    if (!text.trim() || busy.value) return
    if (health.value?.storage_mode === 'multi' && !currentUser.value) {
      notify('请先登录后再对话')
      return
    }
    const conversation = active.value || createConversation()
    if (!retry) conversation.messages.push({ id: crypto.randomUUID(), role: 'user', content: text.trim() })
    if (conversation.messages.filter(item => item.role === 'user').length === 1) conversation.title = text.trim().slice(0, 24)
    conversation.messages.push({ id: crypto.randomUUID(), role: 'assistant', content: '', pending: true, prompt: text, startedAt: Date.now() })
    const message = conversation.messages[conversation.messages.length - 1]
    busy.value = true
    try {
      const body = {
        message: text,
        // multi 模式下后端会忽略该字段，以 JWT 为准；非 multi 仍兼容旧行为。
        user_id: currentUser.value?.user_id || userId,
        conv_id: conversation.serverId || undefined,
        model: selectedModel.value || undefined,
      }
      const response = await api('/chat', { method: 'POST', body })
      conversation.serverId = response.conv_id; message.content = response.response; message.result = response
      // The answer is ready now. Trace and dashboard refreshes are secondary
      // requests and must not keep the reply hidden behind the loading state.
      message.pending = false; busy.value = false
      const loadTrace = api(`/trace/tool/${encodeURIComponent(response.request_id)}`)
        .then(result => { message.trace = result.trace })
        .catch(() => { message.traceError = true })
      void Promise.allSettled([loadTrace, refresh()])
    } catch (error) { message.error = error.message }
    finally { message.pending = false; busy.value = false }
  }

  function exportConversation() { if (active.value) downloadJson(`echomind-conversation-${active.value.id.slice(0, 8)}.json`, active.value) }

  function syncAuth() {
    const next = authUser()
    const prevId = currentUser.value?.user_id || ''
    const nextId = next?.user_id || ''
    currentUser.value = next
    if (!next) {
      // 退出：只隐藏界面，保留账号会话桶，方便同一账号再登录恢复
      hideLocalSessions({ keepBlank: true })
      return
    }
    if (prevId !== nextId || !persistSessions) loadSessionsForUser(next)
  }

  onMounted(() => {
    if (currentUser.value) loadSessionsForUser(currentUser.value)
    else if (!activeId.value) createConversation()
    refresh()
  })

  return {
    view, conversations, activeId, active, health, monitor, skills, notice, busy, connectionError,
    notify, refresh, createConversation, selectConversation, deleteConversation, send, exportConversation,
    catalog, loadModels, selectedModel, agentModels, useModel, currentUser, syncAuth, clearLocalSessions, hideLocalSessions,
  }
}
