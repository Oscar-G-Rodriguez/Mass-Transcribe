if it doesn't work paste this into your code file's terminal

# 1) Install Python 3.12 (if not already installed)
>> winget install -e --id Python.Python.3.12
>>
>> # 2) From your repo folder, create and activate a venv
>> cd C:\Users\Oscar\Documents\GitHub\Mass-Transcribe
>> py -3.12 -m venv .venv
>> .\.venv\Scripts\Activate.ps1
>>
>> # 3) Install backend
>> python -m pip install --upgrade pip
>> python -m pip install faster-whisper
>>
>> # 4) Run with the venv python
>> python .\transcribe.py