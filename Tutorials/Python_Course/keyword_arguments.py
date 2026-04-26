def hello(greeting, title, first, last):
    print(f"{greeting} {title} {first} {last}")

hello("Hello", title = "Mr.",last =  "squarepants", first = "Spongebob")

def get_phone(country, area, first, last):
    return f"{country}-{area}-{first}-{last}"

phone_num = get_phone(country=1, area=123, first=456, last=7890)
print(phone_num)