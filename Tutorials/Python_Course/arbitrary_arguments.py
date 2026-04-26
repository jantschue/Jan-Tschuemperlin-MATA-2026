def display_name(*args):
    for arg in args:
        print(arg, end=" ")


display_name("Dr.","Spongebob","Harold", "Squarepants")


def print_adress(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_adress(street="123 Fake St.",
             apt = "100",
             city="Detroit",
             state="MI",
             zip="54321")


def shipping_label(*args, **kwargs):
    for arg in args:
        print(arg, end=" ")
    print()


    if "apt" in kwargs:
        print(f"{kwargs.get('street')} {kwargs.get('apt')}")
    else:
        print(f"{kwargs.get('street')}")
    print(f"{kwargs.get('city')} {kwargs.get('state')}, {kwargs.get('zip')}")

shipping_label("Dr.","Spongebob","Harold", "Squarepants",
               street="123 Fake Street",
               apt = "#100",
               city ="Detroit",
               state = "MI",
               zip="54321")