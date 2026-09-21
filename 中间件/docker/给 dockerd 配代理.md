---
title: "给 dockerd 配代理"
created: "2026-09-21 10:16:26"
updated: "2026-09-21 10:16:46"
folder: "中间件/docker"
---

``` she
mkdir -p /etc/systemd/system/docker.service.d
cat >/etc/systemd/system/docker.service.d/http-proxy.conf <<'EOF'
[Service]
Environment="HTTP_PROXY=http://192.168.2.83:7897"
Environment="HTTPS_PROXY=http://192.168.2.83:7897"
Environment="NO_PROXY=localhost,127.0.0.1,::1"
EOF

systemctl daemon-reload
systemctl restart docker
docker info | grep -i proxy
```