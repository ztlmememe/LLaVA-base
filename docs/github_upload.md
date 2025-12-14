# 将本地修改推送到 GitHub 的步骤

下面的流程假设仓库已存在本地分支（如当前的 `work` 分支）。如需其他分支名，可将命令中的分支名替换为自己的分支。

## 1. 检查当前修改
- 查看本地未提交的文件：
  ```bash
  git status -sb
  ```
- 如果需要预览差异：
  ```bash
  git diff
  ```

## 2. 配置 Git 身份（首次使用才需要）
如果还未设置过用户名和邮箱，先配置：
```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

## 3. 提交本地修改
- 添加修改：
  ```bash
  git add <文件路径>   # 或使用 git add . 添加全部修改
  ```
- 创建提交：
  ```bash
  git commit -m "描述本次修改的提交信息"
  ```

## 4. 连接 GitHub 仓库（首次推送才需要）
- 如果仓库还没有远程地址，先添加：
  ```bash
  git remote add origin https://github.com/<你的用户名>/<仓库名>.git
  ```
- 也可以使用 SSH 地址：
  ```bash
  git remote add origin git@github.com:<你的用户名>/<仓库名>.git
  ```

## 5. 推送到 GitHub
- 首次推送当前分支：
  ```bash
  git push -u origin <分支名>
  ```
  `-u` 会把本地分支与远程分支关联，后续只需 `git push`。
- 之后的推送：
  ```bash
  git push
  ```

## 6. 处理凭证
- 如果使用 HTTPS 推送，建议在 GitHub 个人设置中创建 **Personal Access Token**，并在首次推送时按提示输入。
- 使用 SSH 时，请确保本地已生成并添加公钥到 GitHub。

## 7. 常见检查
- 查看远程地址：
  ```bash
  git remote -v
  ```
- 查看本地与远程分支关联：
  ```bash
  git branch -vv
  ```

完成以上步骤后，本地修改就会被上传到 GitHub 对应的远程分支。
