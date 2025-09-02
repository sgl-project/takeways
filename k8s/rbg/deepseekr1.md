# test

```

curl -X POST "http://{nodePortIP}:30080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer None" \
  -d '{
        "model": "dsr1",
        "messages": [
          {"role": "system", "content": "0: You are a helpful AI assistant"},
          {"role": "user", "content": "who are you？."}
        ],
        "max_tokens": 221
      }'

```
