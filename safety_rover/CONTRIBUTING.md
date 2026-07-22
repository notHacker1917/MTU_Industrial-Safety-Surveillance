# Contributing to Safety Rover

**Team Collaboration Guide for 3-Person Hackathon Project**

This document outlines branching strategy, Git workflow, code review process, and conflict resolution for the Safety Rover team.

---

## Team Roles & Ownership

| Person | Branch | Owns | Primary Files |
|--------|--------|------|---------------|
| **Person A** | `dev-vision` | Vision pipeline, PPE classifier, dashboard | `oak_pipeline.py`, `tracker_ppe.py`, `ppe_classifier.py`, `dashboard/`, `config/rover_params.yaml` |
| **Person B** | `dev-nav` | SLAM, Nav2, autonomous routing, autonomous control | `ros2_ws/src/navigation_pkg/`, odometry, path planning |
| **Person C** | `dev-presentation` | Orchestration, launch scripts, team coordination, integration | `launch_all.sh`, `setup.sh`, `config_loader.py`, documentation |

---

## Git Branching Strategy

### Main Branches

```
main (production/demo-ready)
  ├─ Protected: Requires 2 approvals before merge
  ├─ Only merge from dev-* branches when fully tested
  ├─ Tag releases: v1.0-demo, v1.1-ppu-improved, etc.
  └─ Always deployable to Pi

dev-vision (Person A's working branch)
  ├─ Merge from: feature branches from A
  ├─ Protected: Person B + C can review
  ├─ Ready to merge to main weekly
  └─ Deploy to Pi for A's testing

dev-nav (Person B's working branch)
  ├─ Merge from: feature branches from B
  ├─ Protected: Person A + C can review
  ├─ Ready to merge to main weekly
  └─ Deploy to Pi for B's testing

dev-presentation (Person C's working branch)
  ├─ Merge from: feature branches from C
  ├─ Protected: Person A + B can review
  ├─ Ready to merge to main weekly
  └─ Deploy to Pi for system testing
```

### Feature Branch Naming

```
dev-vision/feature/shield-heuristic-v2
dev-vision/bugfix/false-alerts-in-low-light
dev-nav/feature/slam-loop-closure
dev-nav/bugfix/odom-drift-correction
dev-presentation/feature/launch-prereq-checks
dev-presentation/docs/team-protocol
```

---

## Workflow: Daily Development

### Setup (Day 1)
```bash
# Clone repo
git clone <url> safety_rover && cd safety_rover

# Create your tracking branch
git checkout -b dev-vision/feature/my-feature
# OR
git checkout -b dev-nav/feature/my-feature
# OR
git checkout -b dev-presentation/feature/my-feature

# Keep in sync with team
git fetch origin
```

### Daily Sync (Before Starting Work)
```bash
# Person A: Keep up with Person B's nav changes
git fetch origin
git merge origin/dev-nav --no-ff  # Integrate nav changes into vision work

# Person B: Keep up with Person A's vision changes
git fetch origin
git merge origin/dev-vision --no-ff  # Integrate vision changes into nav work

# Person C: Integrate both
git fetch origin
git merge origin/dev-vision --no-ff
git merge origin/dev-nav --no-ff
```

### Commit Frequently
```bash
git add .
git commit -m "[vision] Add shield heuristic with HoughLines

- Detects horizontal strap patterns
- Boosts confidence from 0.35–0.65 range to 0.9 if lines detected
- Reduces false negatives in low-light
- Tested with 50 synthetic frames

Fixes: issue #12"
```

### Push & Create Pull Request
```bash
git push origin dev-vision/feature/shield-heuristic-v2

# Then open PR on GitHub/GitLab to dev-vision
# (Don't merge directly; wait for peer review)
```

---

## Code Review Process

### For Vision Changes (Person A pushes to `dev-vision/feature/*`)
1. **Reviewer(s):** Person B + C (async, 12-hour SLA)
2. **Approval Requirements:** ✅ At least 1 approval
3. **Checklist:**
   - [ ] No raw model files committed (`.blob`, `.tflite` should be `.gitignored`)
   - [ ] All functions/classes documented (docstring)
   - [ ] Unit tests added or updated
   - [ ] No hardcoded paths (use `config_loader.py`)
   - [ ] Configuration changes logged in `CHANGELOG.md` (if applicable)
4. **Merge:** Person A can self-merge after approval

**Example Review Comment:**
```
@PersonA Great work on the shield heuristic! A few notes:
1. Line 45: Can you add a comment explaining the 0.35–0.65 range?
2. Shield detection is now 99×99 crop — does that miss small shields?
3. Test coverage looks good; did you verify on the 50-frame synthetic set?

Approving with minor suggestions. Let me know if you'd like to iterate.
```

### For Navigation Changes (Person B pushes to `dev-nav/feature/*`)
1. **Reviewer(s):** Person A + C (async, 12-hour SLA)
2. **Approval Requirements:** ✅ At least 1 approval
3. **Checklist:**
   - [ ] No changes to config files without dual approval
   - [ ] Odometry math validated (test with known distance)
   - [ ] SLAM loop closure tested in test environment
   - [ ] No hardcoded IPs/ports
4. **Merge:** Person B can self-merge after approval

### For Integration/Config Changes (Person C pushes to `dev-presentation/feature/*`)
1. **Reviewer(s):** Person A + B (REQUIRED — not optional)
2. **Approval Requirements:** ✅ Both A **and** B must approve
3. **Checklist:**
   - [ ] `launch_all.sh` tested on fresh Pi image
   - [ ] All node dependencies listed in `package.xml`
   - [ ] No Pi-specific paths in launch files (use env vars)
   - [ ] Documentation updated to match code
4. **Merge:** Person C can merge only after BOTH approvals

---

## Merge Checklist (Before Merging to `dev-*`)

```bash
# 1. Ensure tests pass
pytest tests/ -v

# 2. Check for uncommitted changes
git status  # Should be clean

# 3. Run linter (optional but encouraged)
pylint safety_rover/*.py  # For vision/core
# OR mypy for type checking

# 4. Verify config validity
python -c "from config.config_loader import load_config; load_config()"

# 5. Merge (if using command line)
git checkout dev-vision
git pull origin dev-vision
git merge --no-ff dev-vision/feature/my-feature  # Preserves history
git push origin dev-vision
```

---

## Conflict Resolution

### When Conflicts Happen

```bash
# Typical conflict scenario
git merge origin/dev-nav
# CONFLICT (content): Merge conflict in config/rover_params.yaml

# 1. Open the conflicted file and see markers:
# <<<<<<< HEAD
#   my_ppe_threshold: 0.65
# =======
#   my_ppe_threshold: 0.60
# >>>>>>> origin/dev-nav

# 2. STOP: Don't just pick one. Call team sync.
```

### Config File Conflicts (Highest Priority)

**Rule:** Config changes require **Person C to arbitrate** with input from A + B.

**Process:**
1. Person encountering conflict: post in team chat with diff
2. Person C: meets with A & B async (comments on PR or Discord)
3. Resolution: One person makes the agreed-upon change
4. Re-push & notify team

**Example:**
```
Person A: "Detection confidence: 0.45 helps shield detection"
Person B: "Detection confidence: 0.50 helps reduce false positives in nav"
Person C: "Let's go with 0.48 as compromise; tweak zone rules instead"
```

### Code Conflicts (Lower Priority)

**Rule:** Coder can self-resolve if changes don't touch config.

**Process:**
1. Resolve locally (pick best code, combine if possible)
2. Test locally
3. Commit with message: `Resolved merge conflict from dev-nav`
4. Push & notify team

---

## Git Aliases (Quick Commands)

Add these to your `.gitconfig` for team convenience:

```bash
# Setup (run once)
git config --global alias.sync-vision 'fetch origin && merge origin/dev-vision'
git config --global alias.sync-nav 'fetch origin && merge origin/dev-nav'
git config --global alias.sync-all '!git sync-vision && git sync-nav'

git config --global alias.status-team '!echo "Vision:" && git log --oneline origin/dev-vision -3 && echo "\nNav:" && git log --oneline origin/dev-nav -3'

git config --global alias.pr-status 'log --oneline --decorate --graph --all'
```

**Usage:**
```bash
# Sync up before starting work
git sync-vision

# Check team progress
git status-team

# See all branches & merges
git pr-status
```

---

## Release Process (for Demo)

### Steps to Release v1.0-demo

1. **Team Approval Meeting** (15 min)
   - All features working on dev-* branches?
   - Any critical bugs?
   - Documentation up to date?

2. **Merge to Main**
   ```bash
   # Person C coordinates
   git checkout main
   git pull origin main
   git merge --no-ff dev-vision -m "v1.0-demo: Vision complete"
   git merge --no-ff dev-nav -m "v1.0-demo: Navigation complete"
   git merge --no-ff dev-presentation -m "v1.0-demo: Integration complete"
   ```

3. **Tag Release**
   ```bash
   git tag -a v1.0-demo -m "Safety Rover - Demo Release"
   git push origin main --tags
   ```

4. **Deploy to Pi**
   ```bash
   ssh pi@192.168.1.100
   cd /home/pi/safety_rover
   git checkout v1.0-demo
   bash setup.sh
   bash launch_all.sh
   ```

5. **Test & Document**
   - Run demo_visual.py
   - Test all zones (A, B, Transit)
   - Capture screen/logs
   - Post release notes in README

---

## Communication Channels

| Channel | Purpose | Response Time |
|---------|---------|----------------|
| Discord `#safety-rover` | Quick questions, daily standups | 2 hours |
| PR Comments | Code review, specific feedback | 12 hours (SLA) |
| GitHub Issues | Bug reports, feature requests | 24 hours |
| Weekly Sync | High-level progress, blockers | Friday 3 PM |

---

## Troubleshooting Common Scenarios

### Scenario: "I pushed to main by accident"

```bash
# Undo the push (ask Person C first!)
git reset --soft HEAD~1
git push origin main --force-with-lease  # Recoverable

# Then redo on feature branch
git checkout -b dev-vision/feature/oops
git commit -m "My feature"
git push origin dev-vision/feature/oops
```

### Scenario: "I need Person A's latest shield heuristic for testing"

```bash
# Pull directly from their branch
git fetch origin dev-vision
git merge origin/dev-vision --no-ff

# OR cherry-pick a specific commit
git cherry-pick <commit-hash>
```

### Scenario: "launch_all.sh is broken, need to revert"

```bash
# Find the bad commit
git log --oneline launch_all.sh | head -5

# Revert
git revert <commit-hash>
git push origin dev-presentation

# OR if not yet on main
git reset --hard <last-good-commit>
```

---

## Tips for Success

1. **Commit often** (every 15–30 min) — smaller changes are easier to review
2. **Write descriptive commit messages** — helps team understand intent
3. **Pull before push** — reduces merge conflicts
4. **Test before committing** — `pytest tests/` takes 10 seconds
5. **Review your own PR first** — catch silly mistakes early
6. **Ask for help early** — don't get stuck; tag someone
7. **Keep config changes minimal** — large config diffs are error-prone

---

**Last Updated:** 2026-06-16 | **Version:** 1.0
