#!/bin/bash
# ==============================================================================
# ghostty-terminal-config 一键安装脚本
# ==============================================================================
# 用法:
# bash <(curl -fsSL https://raw.githubusercontent.com/justhalfbit/ghostty-terminal-config/main/install.sh)
#
# 说明:
# 1. 安装 Homebrew 依赖（字体、终端工具、zsh 插件）
# 2. 备份已有配置到 ~/.config-backup/YYYYMMDD\_HHMMSS/
# 3. 从 GitHub 下载配置文件到目标位置
#
# 恢复备份:
# cp ~/.config-backup/<时间戳>/ghostty-config ~/.config/ghostty/config
# cp ~/.config-backup/<时间戳>/starship.toml ~/.config/starship.toml
#
# 卸载 zsh 配置:
# 删除 ~/.zshrc 中 ">>> ghostty-terminal-config >>>" 到 "<<< ghostty-terminal-config <<<" 之间的内容
# ==============================================================================
set -e
REPO\_URL="https://github.com/justhalfbit/ghostty-terminal-config.git"
BACKUP\_DIR="$HOME/.config-backup/$(date +%Y%m%d\_%H%M%S)"
TMP\_DIR="$(mktemp -d)"
# 清理临时目录
cleanup() {
rm -rf "$TMP\_DIR"
}
trap cleanup EXIT
# ==============================================================================
# 用户确认
# ==============================================================================
echo ""
echo "======================================"
echo " Ghostty Terminal Config 安装脚本"
echo "======================================"
echo ""
echo "本脚本将执行以下操作:"
echo " 1. 通过 Homebrew 安装终端工具和字体"
echo " 2. 备份已有 Ghostty 和 Starship 配置到 ~/.config-backup/"
echo " 3. 安装新的终端配置（Ghostty + Starship + zsh）"
echo ""
echo "已有配置将备份到: $BACKUP\_DIR"
echo ""
read -p "是否继续？(y/n) " -n 1 -r < /dev/tty
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
echo "已取消安装。"
exit 0
fi
echo ""
# ==============================================================================
# 检查环境
# ==============================================================================
if ! command -v brew &> /dev/null; then
echo "错误: 未检测到 Homebrew，请先安装: https://brew.sh"
exit 1
fi
if ! command -v git &> /dev/null; then
echo "错误: 未检测到 git，请先安装 Xcode Command Line Tools: xcode-select --install"
exit 1
fi
# ==============================================================================
# 安装依赖
# ==============================================================================
echo "==> 安装 Homebrew 依赖..."
brew install --cask font-maple-mono-nf
if [ ! -d "/Applications/Ghostty.app" ]; then
brew install --cask ghostty
else
echo " Ghostty 已安装，跳过。"
fi
brew install starship fzf zoxide eza bat yazi zsh-autosuggestions zsh-syntax-highlighting zsh-completions
# ==============================================================================
# 下载配置文件
# ==============================================================================
echo "==> 下载配置文件..."
git clone --depth 1 "$REPO\_URL" "$TMP\_DIR/repo"
# ==============================================================================
# 备份已有配置
# ==============================================================================
echo "==> 检查已有配置..."
backup\_file() {
local file="$1"
local name="$2"
if [ -e "$file" ] || [ -L "$file" ]; then
mkdir -p "$BACKUP\_DIR"
if [ -L "$file" ]; then
local target
target="$(readlink "$file")"
echo "$target" > "$BACKUP\_DIR/$name.symlink"
echo " 备份软链接 $file (指向 $target)"
else
cp "$file" "$BACKUP\_DIR/$name"
echo " 备份文件 $file"
fi
fi
}
backup\_file ~/.config/ghostty/config "ghostty-config"
backup\_file ~/.config/starship.toml "starship.toml"
if [ -d "$BACKUP\_DIR" ]; then
echo ""
echo " ✓ 已有配置已备份到: $BACKUP\_DIR"
echo ""
else
echo " 无已有配置，跳过备份。"
fi
# ==============================================================================
# 安装配置文件
# ==============================================================================
echo "==> 安装配置文件..."
mkdir -p ~/.config/ghostty
cp "$TMP\_DIR/repo/ghostty/config" ~/.config/ghostty/config
cp "$TMP\_DIR/repo/starship/starship.toml" ~/.config/starship.toml
echo " ✓ ~/.config/ghostty/config"
echo " ✓ ~/.config/starship.toml"
# .zshrc 追加到用户已有配置尾部（不覆盖）
if grep -q "# >>> ghostty-terminal-config >>>" ~/.zshrc 2>/dev/null; then
echo " .zshrc 中已存在 ghostty-terminal-config 配置，跳过。"
else
{
echo ""
echo "# >>> ghostty-terminal-config >>>"
echo "# 以下内容由 ghostty-terminal-config 安装脚本自动追加"
echo "# 删除方法: 移除从 >>> 到 <<< 之间的所有内容"
cat "$TMP\_DIR/repo/zsh/.zshrc"
echo "# <<< ghostty-terminal-config <<<"
} >> ~/.zshrc
echo " ✓ ~/.zshrc (已追加到尾部)"
fi
# ==============================================================================
# 完成
# ==============================================================================
echo ""
echo "======================================"
echo " 安装完成！"
echo "======================================"
echo ""
echo "请重启 Ghostty 终端生效。"
echo ""
if [ -d "$BACKUP\_DIR" ]; then
echo "恢复旧配置:"
echo " cp $BACKUP\_DIR/ghostty-config ~/.config/ghostty/config"
echo " cp $BACKUP\_DIR/starship.toml ~/.config/starship.toml"
echo ""
fi
echo "卸载 zsh 配置:"
echo " 删除 ~/.zshrc 中 '>>> ghostty-terminal-config >>>' 到 '<<< ghostty-terminal-config <<<' 之间的所有内容"
echo ""
