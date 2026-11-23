# Keep Windows from idle-sleeping while a Mininet campaign runs in WSL (sleep restarts the WSL VM and
# kills the campaign). Uses SetThreadExecutionState: no power settings are changed, and the request
# ends when this script exits (when no experiment process is left in WSL). Closing the lid still sleeps.
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]'0x80000000'; $ES_SYSTEM_REQUIRED = [uint32]'0x00000001'
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
$status = '/mnt/c/Users/User/Downloads/Projects-7/ResilientSDN/docker/emulation_status.sh'
while ($true) {
    Start-Sleep 300
    $out = wsl -d Ubuntu-24.04 -u root -- sh $status 2>$null | Out-String
    if ($out -notmatch 'evaluation\.experiment') { break }
}
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
'campaign finished; wake lock released'
