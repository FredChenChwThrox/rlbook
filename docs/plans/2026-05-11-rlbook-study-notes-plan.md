# RLBook Study Notes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the repository into a study-notes version that preserves structure and key formulas while reducing continuous translation-style prose.

**Architecture:** Update repository-level positioning files first, then rewrite chapter files in place using a consistent note-taking template. Each chapter keeps its numbering and core sections, but long prose is replaced with concise summaries, formula explanations, and learning tips.

**Tech Stack:** Markdown, Git

---

### Task 1: Save planning documents

**Files:**
- Create: `docs/plans/2026-05-11-rlbook-study-notes-design.md`
- Create: `docs/plans/2026-05-11-rlbook-study-notes-plan.md`

**Step 1: Create the design document**

Write the design goals, scope, content rules, and validation criteria.

**Step 2: Create the implementation plan**

Write the execution sequence and file targets.

**Step 3: Verify the files exist**

Run: `Get-ChildItem docs/plans`
Expected: both plan files are listed

### Task 2: Scan chapter structure

**Files:**
- Modify: none

**Step 1: Extract top-level headings from all chapter markdown files**

Run a PowerShell scan to review heading structure.

**Step 2: Define a reusable rewrite pattern**

Use a note-style pattern that preserves headings where practical and shortens paragraphs.

### Task 3: Update repository positioning files

**Files:**
- Modify: `README.md`
- Modify: `NOTICE.md`
- Modify: `COPYRIGHT.md`

**Step 1: Rewrite repository description**

Position the repository as study notes instead of a translation-first repository.

**Step 2: Keep upstream notice visible**

Retain upstream source and citation information.

**Step 3: Keep copyright boundaries explicit**

Make clear that no blanket redistribution rights are granted.

### Task 4: Rewrite chapter files

**Files:**
- Modify: `00-课程介绍.md`
- Modify: `01-基本概念.md`
- Modify: `02-状态值与贝尔曼方程.md`
- Modify: `03-最优状态值与贝尔曼最优方程.md`
- Modify: `04-值迭代与策略迭代.md`
- Modify: `05-蒙特卡洛方法.md`
- Modify: `06-随机逼近.md`
- Modify: `07-时序差分方法.md`
- Modify: `08-值函数方法.md`
- Modify: `09-策略梯度方法.md`
- Modify: `10-Actor-Critic方法.md`
- Modify: `附录-数学预备知识.md`

**Step 1: Keep chapter structure recognizable**

Preserve chapter titles, learning goals, and major section layout where practical.

**Step 2: Compress translation-style prose**

Convert long paragraphs into note-style bullets and short explanations.

**Step 3: Preserve important formulas and pseudocode**

Keep central formulas and add plain-language explanations.

### Task 5: Validate repository state

**Files:**
- Modify: none

**Step 1: Review representative files**

Check README, one early chapter, one algorithm-heavy chapter, and the appendix.

**Step 2: Review git diff**

Run git diff to confirm the repo now reads like study notes.

**Step 3: Commit and push**

Create a documentation commit and push to `origin/main`.
