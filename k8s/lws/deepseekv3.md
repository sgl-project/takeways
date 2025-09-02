# Kubernetes 部署指南

本项目包含以下 Kubernetes 资源文件：

* **p.yaml**\
  定义 **Prefill 节点**（负责大模型推理的前向预填充部分）。
* **d.yaml**\
  定义 **Decode 节点**（负责大模型推理的解码阶段）。
* **lws-router.yaml**\
  定义 **Rust Router**，用于请求转发和流量调度。
* **rbac.yaml**\
  定义 **RBAC 权限**（ServiceAccount、Role、RoleBinding），保证组件能访问所需的 Kubernetes 资源。
* **svc.yaml**\
  定义 **Service**，用于服务暴露与集群内服务发现。

***

## 创建资源

依次执行以下命令：

```bash
# 创建 RBAC 权限
kubectl apply -f rbac.yaml

# 创建 Service
kubectl apply -f svc.yaml

# 创建 Prefill 节点
kubectl apply -f p.yaml

# 创建 Decode 节点
kubectl apply -f d.yaml

# 创建 Router
kubectl apply -f lws-router.yaml

删除资源

如果需要清理部署的所有资源：

kubectl delete -f lws-router.yaml
kubectl delete -f d.yaml
kubectl delete -f p.yaml
kubectl delete -f svc.yaml
kubectl delete -f rbac.yaml

查询状态
查看 Pod
kubectl get pods -o wide

查看 Service
kubectl get svc

查看 Deployment / DaemonSet
kubectl get deploy
kubectl get ds

查看日志
# Router 日志
kubectl logs -f <router-pod-name>
如: kubectl logs lws-router-5b8ddc5645-jzsm4 -naiservice

# Prefill 节点日志
kubectl logs -f <prefill-pod-name>

# Decode 节点日志
kubectl logs -f <decode-pod-name>

滚动更新

修改 YAML 后，可以重新应用：

kubectl apply -f p.yaml
kubectl apply -f d.yaml
kubectl apply -f lws-router.yaml


或强制重启 Deployment：

kubectl rollout restart deploy <deployment-name>

命名空间（可选）

如果要将资源放在指定命名空间（例如 aiservice），请在 YAML 文件中增加：

metadata:
  namespace: aiservice


查询时需要加上 -n 参数：

kubectl get pods -n aiservice


## 服务访问方式

svc.yaml 使用了 NodePort 类型，端口号为 31080。
这意味着可以通过 任意 Kubernetes 节点的 IP 来访问服务：

http://<node-ip>:31080/v1/chat/completions

例如：

curl -X POST "http://10.155.83.42:31080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer None" \
  -d '{
        "model": "qwen2",
        "messages": [
          {"role": "system", "content": "0: You are a helpful AI assistant"},
          {"role": "user", "content": "你是谁？."}
        ],
        "max_tokens": 221
      }'



```
