## Navigation Menu

# Search code, repositories, users, issues, pull requests...

# Provide feedback

We read every piece of feedback, and take your input very seriously.

# Saved searches

## Use saved searches to filter your results more quickly

To see all available qualifiers, see our [documentation](https://docs.github.com/search-github/github-code-search/understanding-github-code-search-syntax).

# justhalfbit/ghostty-terminal-config

## Folders and files

| Name | | Name | Last commit message | Last commit date |
| --- | --- | --- | --- | --- |
| Latest commit   History[15 Commits](/justhalfbit/ghostty-terminal-config/commits/main/)   15 Commits | | |
| [ghostty](/justhalfbit/ghostty-terminal-config/tree/main/ghostty "ghostty") | | [ghostty](/justhalfbit/ghostty-terminal-config/tree/main/ghostty "ghostty") |  |  |
| [starship](/justhalfbit/ghostty-terminal-config/tree/main/starship "starship") | | [starship](/justhalfbit/ghostty-terminal-config/tree/main/starship "starship") |  |  |
| [zsh](/justhalfbit/ghostty-terminal-config/tree/main/zsh "zsh") | | [zsh](/justhalfbit/ghostty-terminal-config/tree/main/zsh "zsh") |  |  |
| [LICENSE](/justhalfbit/ghostty-terminal-config/blob/main/LICENSE "LICENSE") | | [LICENSE](/justhalfbit/ghostty-terminal-config/blob/main/LICENSE "LICENSE") |  |  |
| [README.md](/justhalfbit/ghostty-terminal-config/blob/main/README.md "README.md") | | [README.md](/justhalfbit/ghostty-terminal-config/blob/main/README.md "README.md") |  |  |
| [install.sh](/justhalfbit/ghostty-terminal-config/blob/main/install.sh "install.sh") | | [install.sh](/justhalfbit/ghostty-terminal-config/blob/main/install.sh "install.sh") |  |  |
| View all files | | |

## Latest commit

## History

## Repository files navigation

# Ghostty Terminal Config

macOS 下基于 Ghostty + Starship + zsh 插件的终端美化方案，从 iTerm2 + oh-my-zsh 迁移而来，更轻量更快。

## 效果

## 包含的配置文件

| 文件 | 说明 | 安装位置 |
| --- | --- | --- |
| `ghostty/config` | Ghostty 终端配置（字体、主题、窗口、光标） | `~/.config/ghostty/config` |
| `starship/starship.toml` | Starship 彩虹条提示符配置（官方预设 + 换行） | `~/.config/starship.toml` |
| `zsh/.zshrc` | zsh 配置（插件、工具、别名、快捷键） | `~/.zshrc` |

`ghostty/config`
`~/.config/ghostty/config`
`starship/starship.toml`
`~/.config/starship.toml`
`zsh/.zshrc`
`~/.zshrc`

## 一键安装

安装前会询问确认，确认后自动执行：

`~/.zshrc`

## 备份与恢复

安装脚本会自动将已有配置备份到 `~/.config-backup/<时间戳>/` 目录。

`~/.config-backup/<时间戳>/`

备份文件对应关系：

| 备份文件 | 原始位置 |
| --- | --- |
| `~/.config-backup/<时间戳>/ghostty-config` | `~/.config/ghostty/config` |
| `~/.config-backup/<时间戳>/starship.toml` | `~/.config/starship.toml` |

`~/.config-backup/<时间戳>/ghostty-config`
`~/.config/ghostty/config`
`~/.config-backup/<时间戳>/starship.toml`
`~/.config/starship.toml`

恢复命令：

### 卸载 zsh 配置

删除 `~/.zshrc` 中 `# >>> ghostty-terminal-config >>>` 到 `# <<< ghostty-terminal-config <<<` 之间的所有内容即可。

`~/.zshrc`
`# >>> ghostty-terminal-config >>>`
`# <<< ghostty-terminal-config <<<`

## 依赖

| 工具 | 用途 |
| --- | --- |
| [Ghostty](https://ghostty.org) | GPU 加速终端模拟器 |
| [Starship](https://starship.rs) | 跨 shell 提示符 |
| [fzf](https://github.com/junegunn/fzf) | 模糊搜索（Ctrl+R 搜历史，Ctrl+T 搜文件） |
| [zoxide](https://github.com/ajeetdsouza/zoxide) | 智能目录跳转（`z foo` 代替 `cd`） |
| [eza](https://github.com/eza-community/eza) | 替代 ls，彩色图标 |
| [bat](https://github.com/sharkdp/bat) | 替代 cat，语法高亮 |
| [yazi](https://github.com/sxyazi/yazi) | 终端文件管理器 |
| [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions) | 历史命令自动建议 |
| [zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting) | 命令语法高亮 |
| [zsh-completions](https://github.com/zsh-users/zsh-completions) | Tab 补全增强 |
| [Maple Mono NF](https://github.com/subframe7536/maple-font) | 终端字体（Nerd Font，中文显示优秀） |

`z foo`
`cd`

## 手动安装

### 1. 安装依赖

### 2. 下载配置文件

### 3. 安装配置文件

注意：zsh 配置是追加到 `~/.zshrc` 尾部，不会覆盖已有内容。如果重复执行需手动去重。

`~/.zshrc`

### 4. 清理并重启

重启 Ghostty 终端生效。

## Starship 预设说明

彩虹条基于 `starship preset catppuccin-powerline` 官方预设，唯一改动：

`starship preset catppuccin-powerline`
`[line_break] disabled = false`

## 快捷键速查

| 快捷键 | 功能 |
| --- | --- |
| `Ctrl+F` | 接受自动建议 |
| `Ctrl+R` | fzf 模糊搜索历史命令 |
| `Ctrl+T` | fzf 模糊搜索文件 |
| `Tab` | 补全，连续按在候选列表中移动 |

`Ctrl+F`
`Ctrl+R`
`Ctrl+T`
`Tab`

## 别名速查

| 别名 | 实际命令 |
| --- | --- |
| `ls` | `eza --icons --group-directories-first` |
| `ll` | `eza -l --icons --sort=name` |
| `lt` | `eza --tree --icons --level=2` |
| `cat` | `bat --paging=never --style=plain` |
| `y` | yazi 文件管理器（退出自动 cd） |
| `z foo` | zoxide 智能跳转到包含 foo 的目录 |

`ls`
`eza --icons --group-directories-first`
`ll`
`eza -l --icons --sort=name`
`lt`
`eza --tree --icons --level=2`
`cat`
`bat --paging=never --style=plain`
`y`
`z foo`

## About

macOS 终端美化配置：Ghostty + Starship + zsh 插件（从 iTerm2 + oh-my-zsh 迁移的轻量方案）

### Resources

### License

### Uh oh!

There was an error while loading. Please reload this page.

There was an error while loading. Please reload this page.

### Stars

### Watchers

### Forks

## [Releases](/justhalfbit/ghostty-terminal-config/releases)

## [Packages 0](/users/justhalfbit/packages?repo_name=ghostty-terminal-config)

### Uh oh!

There was an error while loading. Please reload this page.

There was an error while loading. Please reload this page.

## [Contributors](/justhalfbit/ghostty-terminal-config/graphs/contributors)

### Uh oh!

There was an error while loading. Please reload this page.

There was an error while loading. Please reload this page.

## Languages

## Footer

### Footer navigation
