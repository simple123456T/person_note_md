---
title: "spark重置密码"
created: "2026-09-09 10:42:19"
updated: "2026-09-09 10:42:40"
folder: "教程"
---

```
① 关机

② 开机

③ 开机后立即连续按：
   左 Shift + Insert

④ 进入 GRUB

⑤ 选：
   DGX OS GNU/Linux

⑥ 按：
   e

⑦ 找到 linux 那一行 linux /boot/vmlinuz-xxxxx root=UUID=xxxx ro ... $vt_handoff

⑧ 把：
   ro

   改成：
   rw

⑨ 在这一行最后：
   $vt_handoff

   后面添加：

   init=/bin/bash console=tty0

   最终类似：

   linux /boot/vmlinuz-xxxxx root=UUID=xxxxx rw ... $vt_handoff init=/bin/bash console=tty0

⑩ 确保全部在同一行

⑪ Ctrl + X 启动 或者 F10

⑫ 看到：
   root@(none):/#

⑬ 执行：

   mount -o remount,rw /

⑭ 执行：

   passwd 你的用户名

⑮ 输入新密码两次

⑯ 执行：

   exec /sbin/init

   或直接断电重启

⑰ 正常登录 DGX Spark



```