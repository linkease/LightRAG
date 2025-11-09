# X-Openai-Key Header Feature - Implementation Summary

## Overview

This implementation adds support for passing user-specific OpenAI API keys via HTTP headers to LightRAG API endpoints. This enables multi-tenant applications to track token usage per user.

## Changes Made

### 1. Core Data Model (`lightrag/base.py`)
- Added `user_api_key` field to `QueryParam` class
- Allows passing user-specific API keys through the query pipeline

### 2. API Routes (`lightrag/api/routers/query_routes.py`)
- Added `x_openai_key` header parameter to all query endpoints:
  - `POST /query` - Standard non-streaming query
  - `POST /query/stream` - Streaming query
  - `POST /query/data` - Data retrieval only
- Extracts `X-Openai-Key` header and passes it to `QueryParam.user_api_key`
- Logs when user-specific API key is used

### 3. LightRAG Core (`lightrag/lightrag.py`)
- Modified `aquery_llm()` to pass `user_api_key` to `global_config`
- Modified `aquery_data()` to pass `user_api_key` to `global_config`
- Ensures user API key is available throughout the query pipeline

### 4. LLM Bindings
#### OpenAI (`lightrag/llm/openai.py`)
- Modified `openai_complete_if_cache()` to prioritize `user_api_key`
- API key priority: `user_api_key` > `api_key` parameter > environment variable
- Logs when user-specific API key is being used

#### Azure OpenAI (`lightrag/llm/azure_openai.py`)
- Added same `user_api_key` support for Azure OpenAI
- Also added `llm_query_model` support (was missing)
- Maintains consistency with standard OpenAI implementation

### 5. Documentation
- Created comprehensive documentation: `docs/X-Openai-Key-Header.md`
- Includes usage examples for Python, JavaScript/TypeScript, and cURL
- Covers multi-tenant scenarios and security best practices

### 6. Testing
- Created test script: `test_user_api_key.py`
- Tests all three query endpoints with and without custom API key
- Validates header handling and fallback behavior

## Usage Example

```python
import requests

# Query with user-specific API key
response = requests.post(
    "http://localhost:8020/query",
    headers={
        "Content-Type": "application/json",
        "X-Openai-Key": "sk-user-123-specific-key"
    },
    json={
        "query": "What is machine learning?",
        "mode": "mix"
    }
)
```

## API Key Priority

The system uses API keys in the following priority order:

1. **X-Openai-Key Header** (highest priority)
2. **api_key function parameter**
3. **OPENAI_API_KEY environment variable** (fallback)

## Benefits

1. **Per-User Billing**: Each user can use their own OpenAI API key
2. **Token Tracking**: Easily track token usage per user
3. **Multi-Tenant Support**: Different users can have different API quotas
4. **Flexible Access Control**: Different access tiers can use different keys

## Security Considerations

- Always use HTTPS in production to protect API keys in transit
- API keys are not logged in full (only partial logging for debugging)
- Keys are handled in memory only and not persisted

## Testing

Run the test script to verify functionality:

```bash
python test_user_api_key.py
```

Check server logs for these messages to confirm it's working:
- `INFO: Using user-specific API key from X-Openai-Key header`
- `DEBUG: Using user-specific API key for this request`

## Backward Compatibility

This feature is fully backward compatible:
- If `X-Openai-Key` header is not provided, the system uses the default API key
- Existing code continues to work without any changes
- No breaking changes to API contracts

## Files Modified

1. `lightrag/base.py` - Added `user_api_key` to `QueryParam`
2. `lightrag/api/routers/query_routes.py` - Added header parameter to endpoints
3. `lightrag/lightrag.py` - Pass `user_api_key` through `global_config`
4. `lightrag/llm/openai.py` - Prioritize `user_api_key` in API calls
5. `lightrag/llm/azure_openai.py` - Added `user_api_key` support

## Files Added

1. `docs/X-Openai-Key-Header.md` - Comprehensive documentation
2. `test_user_api_key.py` - Test script
3. `docs/IMPLEMENTATION_X_OPENAI_KEY.md` - This file

## Future Enhancements

Potential improvements for future versions:
1. Add support for other LLM providers (Anthropic, Google, etc.)
2. Add API key validation before making LLM calls
3. Add metrics/logging for API key usage patterns
4. Support for API key rotation without service restart
