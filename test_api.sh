#!/bin/bash

# =============================================================================
# IndexTTS API 新接口测试脚本
# =============================================================================
# 
# 使用前请先:
# 1. 启动 IndexTTS API 服务
# 2. 将下面的 NGROK_URL 替换为实际的 ngrok 公开链接
# 3. 运行脚本: bash test_api.sh
#

# 配置部分 - 请替换为你的实际 ngrok URL
NGROK_URL="https://your-ngrok-url.ngrok.io"

# 测试参数
VOICE_URL="https://video.allinaiimage.com/sample1.wav"
TEST_TEXT="这是一段测试文本，用于验证新的 API 接口是否正常工作。我们使用网络音频链接作为音色参考。"

# API 端点
API_ENDPOINT="${NGROK_URL}/api/synthesize_with_storage"

echo "=========================================="
echo "🚀 测试 IndexTTS API 新接口"
echo "=========================================="
echo "API 地址: ${API_ENDPOINT}"
echo "音色链接: ${VOICE_URL}"
echo "测试文本: ${TEST_TEXT}"
echo ""
echo "开始发送请求..."
echo "=========================================="
echo ""

# 发送 POST 请求
curl -X POST "${API_ENDPOINT}" \
  -F "voice_path=${VOICE_URL}" \
  -F "text=${TEST_TEXT}" \
  -H "Accept: application/json" \
  -v

echo ""
echo ""
echo "=========================================="
echo "✅ 请求已发送"
echo "=========================================="
