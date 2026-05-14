"""Unit tests for plaud-mcp transcription tools (cmd_905 wave1)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from plaud_mcp.server import (
    check_connection,
    create_folder,
    get_file,
    get_file_count,
    get_files,
    get_recent_files,
    get_summary,
    get_task_status,
    get_transcript,
    list_folders,
    move_file_to_folder,
    search_transcripts,
    start_transcription,
    unset_file_folder,
)

EXPECTED_TRANSCRIPTION_BODY = {
    "is_reload": 0,
    "summ_type": "AUTO-SELECT",
    "summ_type_type": "system",
    "info": '{"language":"auto","timezone":9,"diarization":1,"llm":"auto"}',
    "support_mul_summ": True,
}


@pytest.fixture
def mock_client():
    """Return a mock PlaudClient context manager."""
    with patch("plaud_mcp.server.PlaudClient") as mock_client_cls:
        instance = mock_client_cls.return_value.__aenter__.return_value
        yield mock_client_cls, instance


@pytest.mark.asyncio
async def test_start_transcription_dry_run_does_not_post(mock_client):
    mock_client_cls, instance = mock_client
    result = await start_transcription(file_id=" file-abc ", dry_run=True)

    assert result == {
        "ok": True,
        "dry_run": True,
        "file_id": "file-abc",
        "method": "POST",
        "endpoint": "/ai/transsumm/file-abc",
        "body": EXPECTED_TRANSCRIPTION_BODY,
        "message": "Planned: start transcription for file file-abc",
    }
    mock_client_cls.assert_not_called()
    instance.post.assert_not_called()


@pytest.mark.asyncio
async def test_start_transcription_dry_run_body_varies_only_is_reload(mock_client):
    mock_client_cls, _instance = mock_client
    result = await start_transcription(file_id="file-abc", is_reload=1)

    expected_body = {**EXPECTED_TRANSCRIPTION_BODY, "is_reload": 1}
    assert result["body"] == expected_body
    assert set(result["body"].keys()) == set(EXPECTED_TRANSCRIPTION_BODY.keys())
    mock_client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_start_transcription_empty_file_id_rejected(mock_client):
    mock_client_cls, instance = mock_client
    result = await start_transcription(file_id="  ")

    assert result["ok"] is False
    assert result["errors"] == "empty file_id"
    mock_client_cls.assert_not_called()
    instance.post.assert_not_called()


@pytest.mark.asyncio
async def test_start_transcription_live_posts_with_existing_client(mock_client):
    _mock_client_cls, instance = mock_client
    instance.post.return_value = {"status": 0, "data": {"task_id": "task-1"}}

    result = await start_transcription(
        file_id="file-abc",
        is_reload=1,
        dry_run=False,
    )

    expected_body = {**EXPECTED_TRANSCRIPTION_BODY, "is_reload": 1}
    instance.post.assert_called_once_with("/ai/transsumm/file-abc", json=expected_body)
    assert result["ok"] is True
    assert result["dry_run"] is False
    assert result["response"] == {"status": 0, "data": {"task_id": "task-1"}}


@pytest.mark.asyncio
async def test_get_task_status_uses_existing_client_get(mock_client):
    _mock_client_cls, instance = mock_client
    instance.get.return_value = {"status": 0, "data": {"running": []}}

    result = await get_task_status()

    instance.get.assert_called_once_with("/ai/file-task-status")
    assert result == {"status": 0, "data": {"running": []}}


def test_existing_tool_functions_still_importable():
    expected_tools = [
        check_connection,
        get_file_count,
        get_recent_files,
        get_files,
        get_file,
        get_transcript,
        get_summary,
        search_transcripts,
        list_folders,
        create_folder,
        move_file_to_folder,
        unset_file_folder,
        start_transcription,
        get_task_status,
    ]
    assert all(callable(tool) for tool in expected_tools)
