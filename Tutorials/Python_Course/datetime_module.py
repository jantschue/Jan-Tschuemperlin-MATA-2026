import datetime

date = datetime.date(2025, 1, 2)
today = datetime.date.today()

time = datetime.time(12, 30, 0)
now = datetime.datetime.now()

now = now.strftime("%H:%M:%S %m-%d-%Y")

target = datetime.datetime(2030, 2,1, 20, 30,1)
current = datetime.datetime.now()

if target < current:
    print("Target date has passed")
else:
    print("Target date has not passed")