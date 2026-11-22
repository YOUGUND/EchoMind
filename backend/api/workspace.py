"""Workspace routes for browsing the knowledge collection."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from api.auth import require_user_if_multi
from api.user_context import AuthUser, bind_request_user

router = APIRouter(prefix='/knowledge', tags=['Workspace'])


def knowledge(request: Request):
    kb = getattr(request.app.state, 'knowledge', None)
    if kb is None:
        raise HTTPException(503, '知识空间尚未初始化')
    return kb


@router.get('/documents')
async def documents(
    request: Request,
    user: Optional[AuthUser] = Depends(require_user_if_multi),
):
    kb = knowledge(request)
    if not hasattr(kb, 'list_documents'):
        raise HTTPException(501, '当前存储模式尚未支持文档管理')
    with bind_request_user(user):
        if user is not None and hasattr(kb, 'ensure_default_docs_async'):
            await kb.ensure_default_docs_async(user.id, force=False)
        if user is not None:
            items = await kb.list_documents_async(user_id=user.id) if hasattr(kb, 'list_documents_async') else kb.list_documents(user_id=user.id)
        else:
            items = kb.list_documents()
    return {'items': items}


@router.delete('/documents/{doc_id}')
async def delete_document(
    doc_id: str,
    request: Request,
    user: Optional[AuthUser] = Depends(require_user_if_multi),
):
    kb = knowledge(request)
    if not hasattr(kb, 'delete_document'):
        raise HTTPException(501, '当前存储模式尚未支持文档管理')
    with bind_request_user(user):
        if user is not None and hasattr(kb, 'delete_document_async'):
            removed = await kb.delete_document_async(doc_id, user_id=user.id)
        elif user is not None:
            removed = kb.delete_document(doc_id, user_id=user.id)
        else:
            removed = kb.delete_document(doc_id)
    if not removed:
        raise HTTPException(404, '文档已不存在')
    manager = request.app.state.tool_manager
    manager.clear_cache()
    return {'removed_chunks': removed}
