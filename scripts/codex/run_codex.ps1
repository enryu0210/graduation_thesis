<#
.SYNOPSIS
    Codex 호출 래퍼 — docs/handoff/TASK.md 를 지시서로 넘기고 RESULT.md 를 회수한다.

.DESCRIPTION
    Claude(계획·문서) ↔ Codex(구현·검증) 분업의 2단계를 담당한다.
    맨손 `codex exec` 대비 이 래퍼가 해주는 것:
      - PYTHONIOENCODING=utf-8 주입 (한글 콘솔 cp949 에서 파이썬이 죽는 문제)
      - 경로 기본값과 사전 점검 (TASK.md 존재 / git 상태 / GPU 유무)
      - RESULT.md 갱신 여부 확인 — Codex 가 보고를 빠뜨리면 잡아낸다

.PARAMETER ReadOnly
    read-only 샌드박스로 실행한다. 코드 수정이 불가능하므로 "검증만 재실행"에 안전하다.
    RESULT.md 가 미덥지 않을 때 이 모드로 다시 부른다.

.PARAMETER FullAccess
    danger-full-access 샌드박스. 기본 workspace-write 는 아웃바운드가 막혀 있어서
    pip install 등 네트워크가 필요할 때만 명시적으로 쓴다.

.EXAMPLE
    powershell -File scripts/codex/run_codex.ps1
.EXAMPLE
    powershell -File scripts/codex/run_codex.ps1 -ReadOnly
#>
[CmdletBinding()]
param(
    [switch]$ReadOnly,
    [switch]$FullAccess,
    [string]$Task   = 'docs/handoff/TASK.md',
    [string]$Result = 'docs/handoff/RESULT.md',
    [string]$Model
)

$ErrorActionPreference = 'Stop'

# 콘솔을 UTF-8 로 — 한글 지시서/보고가 깨지지 않게 한다.
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

# 저장소 루트는 이 스크립트 위치(scripts/codex)에서 두 단계 위로 고정한다.
# 호출자의 현재 디렉터리에 의존하면 어디서 부르냐에 따라 동작이 달라진다.
$RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$TaskPath   = Join-Path $RepoRoot $Task
$ResultPath = Join-Path $RepoRoot $Result
$LastMsg    = Join-Path $RepoRoot 'docs/handoff/LAST_MESSAGE.md'

function Write-Step($msg) { Write-Host "[run_codex] $msg" -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host "[run_codex] ⚠ $msg" -ForegroundColor Yellow }

# 사전 점검 실패용 — 스택 트레이스 없이 원인만 보여주고 끝낸다.
function Fail($msg) {
    Write-Host "[run_codex] ✗ $msg" -ForegroundColor Red
    exit 1
}

# ── 사전 점검 ────────────────────────────────────────────────────────────────
if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    Fail "codex CLI 를 찾을 수 없습니다. `npm i -g @openai/codex` 후 다시 시도하세요."
}
if (-not (Test-Path $TaskPath)) {
    Fail "지시서가 없습니다: $TaskPath`n   docs/handoff/TASK.template.md 를 복사해 먼저 작성하세요."
}

$prompt = Get-Content -Path $TaskPath -Raw -Encoding UTF8
if ([string]::IsNullOrWhiteSpace($prompt)) {
    Fail "지시서가 비어 있습니다: $TaskPath"
}
# 명령줄 인자로 넘기므로 Windows 상한(32767자)에 걸리지 않는지 확인한다.
# 넘치면 인자가 조용히 잘려 Codex 가 지시 일부만 보고 작업하게 된다 — 가장 나쁜 실패다.
if ($prompt.Length -gt 30000) {
    Fail "지시서가 너무 깁니다($($prompt.Length)자). 30000자 이하로 줄이거나 stdin 방식으로 직접 호출하세요:`n   codex exec -C '$RepoRoot' -s workspace-write - < $Task"
}

# 샌드박스 결정 — ReadOnly 가 FullAccess 보다 우선한다(안전한 쪽으로 넘어짐).
$sandbox = 'workspace-write'
if ($FullAccess) { $sandbox = 'danger-full-access' }
if ($ReadOnly)   { $sandbox = 'read-only' }

# ── 환경 안내 ────────────────────────────────────────────────────────────────
# 파이썬 비-ASCII 출력이 cp949 콘솔에서 UnicodeEncodeError 로 죽는 것을 막는다.
$env:PYTHONIOENCODING = 'utf-8'

Write-Step "저장소   : $RepoRoot"
Write-Step "지시서   : $Task ($($prompt.Length)자)"
Write-Step "샌드박스 : $sandbox"

# GPU 유무는 "학습을 시킬 수 있는 작업인가"를 가른다. 노트북이면 미리 알려준다.
$cuda = & python -c "import torch;print(torch.cuda.is_available())" 2>$null
if ($cuda -match 'True') {
    Write-Step "GPU      : 사용 가능 (데스크톱) — 학습 작업 가능. 단 GPU 는 1대뿐이니 순차 실행할 것"
} else {
    Write-Warn "GPU      : 없음 (노트북 추정) — 학습이 필요한 작업이면 검증 불가로 보고될 것입니다"
}

# 커밋되지 않은 변경이 있으면, Codex 의 변경과 섞여 나중에 구분이 안 된다.
$dirty = & git -C $RepoRoot status --porcelain
if ($dirty) {
    $n = ($dirty | Measure-Object -Line).Lines
    Write-Warn "커밋되지 않은 변경 $n 건이 있습니다. Codex 변경과 섞이면 나중에 구분이 어렵습니다."
}

# RESULT.md 가 실제로 새로 쓰였는지 판정하기 위해 이전 시각을 기록해 둔다.
$before = $null
if (Test-Path $ResultPath) { $before = (Get-Item $ResultPath).LastWriteTimeUtc }

# ── 호출 ─────────────────────────────────────────────────────────────────────
# read-only 에서는 Codex 가 RESULT.md 를 쓸 수 없다(샌드박스가 막는다).
# 대신 -o 는 CLI 자신이 쓰므로 통과한다 → 보고를 그리로 직접 받는다.
# 쓰기 모드에서는 Codex 가 RESULT.md 를 직접 쓰고, -o 는 안전망으로만 둔다.
$outPath = $LastMsg
if ($ReadOnly) {
    $outPath = $ResultPath
    $prompt = "⚠️ 이 실행은 read-only 샌드박스라 어떤 파일도 쓸 수 없습니다(RESULT.md 포함).`n" +
              "보고 전문을 **최종 메시지**에 담으세요 — 그 메시지가 그대로 RESULT.md 로 저장됩니다.`n`n" +
              $prompt
}

$codexArgs = @('exec', '-C', $RepoRoot, '-s', $sandbox, '-o', $outPath)
if ($Model) { $codexArgs += @('-m', $Model) }
$codexArgs += $prompt

Write-Step "Codex 호출 중… (Ctrl+C 로 중단)"
$sw = [Diagnostics.Stopwatch]::StartNew()
& codex @codexArgs
$exit = $LASTEXITCODE
$sw.Stop()

Write-Step "종료 코드 $exit / 소요 $([int]$sw.Elapsed.TotalSeconds)초"

# ── 사후 점검 ────────────────────────────────────────────────────────────────
if ($exit -ne 0) {
    Write-Warn "Codex 가 0 이 아닌 코드로 종료했습니다. RESULT.md 를 신뢰하지 마세요."
}

$after = $null
if (Test-Path $ResultPath) { $after = (Get-Item $ResultPath).LastWriteTimeUtc }

if ($null -eq $after) {
    Write-Warn "$Result 가 없습니다 — 보고가 회수되지 않았습니다."
    if (-not $ReadOnly) {
        Write-Warn "최종 메시지는 docs/handoff/LAST_MESSAGE.md 에 남아 있으니 그것으로 확인하세요."
    }
} elseif ($before -eq $after) {
    Write-Warn "$Result 가 갱신되지 않았습니다(이전 작업의 잔재일 수 있음). 파일 날짜를 확인하세요."
} else {
    Write-Step "보고서 : $Result (갱신됨)"
    if ($ReadOnly) {
        Write-Step "         └ read-only 모드라 Codex 의 최종 메시지를 그대로 받은 것입니다"
    }
}

Write-Host ''
Write-Step 'RESULT.md 를 읽을 때 확인할 것 — 실제 명령 출력이 붙어 있는가 /'
Write-Step '"미검증" 절이 정말 비었는가 / 수용 기준을 항목별로 대조했는가 / 테스트 168개가 다 돌았는가'

exit $exit
