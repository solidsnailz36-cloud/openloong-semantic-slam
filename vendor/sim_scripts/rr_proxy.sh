#!/usr/bin/env bash
# rr_proxy.sh "<remote command>"  -- same as rr.sh, but tunnel through a local
# SOCKS5 proxy.  Use when the direct campus-network path to the box is broken.
#
#   默认走 127.0.0.1:10808（v2rayN 的 SOCKS 入口）
#   换一个:  SOCKS=127.0.0.1:10809 bash rr_proxy.sh 'hostname'
#
# ⚠️ 前提：代理必须是【全局模式】。规则/PAC 模式下，国内目的地会被放直连，
#    于是又回到那条不通的路 —— 表现是 SOCKS 握手成功但读不到 SSH banner。
set +e
export MSYS2_ARG_CONV_EXCL='*'
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -W 2>/dev/null || pwd)"
[ -f "$HERE/rr.sh" ] || HERE="$HERE/../../.workbuddy/tmp"
export SSH_SOCKS="${SOCKS:-127.0.0.1:10808}"
exec bash "$HERE/rr.sh" "$1"
