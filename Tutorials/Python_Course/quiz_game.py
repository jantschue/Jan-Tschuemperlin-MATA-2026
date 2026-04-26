questions = ("How many elements?: ",
             "Largest eggs?: ",
             "Most abundant gas?: ",
             "Bones?: ",
             "Hottest Planet?: ")

options = (("A. 116 ", "B. 117 ", "C. 118 ", "D. 119 "),
           ("A. Whale ", "B. Crocodile ", "C. Elephant ", "D. Ostrich "),
           ("A. Nitrogen ", "B. Oxygen ", "C. Carbon-Dioxide ", "D. Hydrogen  "),
           ("A. 206 ", "B. 207 ", "C. 208 ", "D. 209 "),
           ("A. Mercury ", "B. Venus ", "C. Earth ", "D. Mars "))

answer = ("C", "D", "A", "A", "B")
guesses = []
score = 0
question_num = 0

for question in questions:
    print("-----------------")
    print(question)
    for option in options[question_num]:
        print(option)

    guess = input("Enter (A, B, C, D): ").upper()
    guesses.append(guess)
    if guess == answer[question_num]:
        score += 1
        print("Correct")
    else:
        print("Incorrect")
        print(f"{answer[question_num]} is the correct answer")

    question_num += 1

print("Answers: ", end =" ")
for answer in answer:
    print(answer, end = " ")
print()

print("Guesses: ", end =" ")
for guess in guesses:
    print(guess, end = " ")
print()

score = int(score / len(questions) * 100)
print(f"{score}%")