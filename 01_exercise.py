# run py script through terminal with '/bin/python3'
# Eric Brown

# Ex 1. Print
print('Hello World')
print()

# Ex 2. input, f-string
num1 = input("Enter first number: ")
num2 = input("Enter second number: ")
print(num1, ' * ', num2, ' = ', str(int(num1)*int(num2)))
print()

# Ex 3. import, random
import random

num1 = random.randrange(100)
num2 = random.randrange(100)
print(str(num1), ' * ', str(num2), ' = ', str(num1*num2))
print()

# Ex 4. loops
for i in range(1, 11):
    for j in range(1, 11):
        print(str(j*i).rjust(4), end='')
    print('')
print()

# Ex 5. functions, if..else
def compare(a, b):
    if a > b:
        print('a is greater')
    elif b > a:
        print('b is greater')
    else:
        print('equal')

num1 = input("Enter first number (a): ")
num2 = input("Enter second number (b): ")

compare(int(num1), int(num2))
print()

# Ex 6
val = random.randrange(100)
inp = input("Guess the number I'm thinking (1-100): ")

while int(inp) != val:
    if val > int(inp):
        print("guess higher")
    else:
        print("guess lower")
    inp = input("Guess the number I'm thinking (1-100): ")

print(inp, " is correct!")
print()

# Ex 7
print(" MULTIPLICATION TEST ")
cnt = 0
for i in range(5):
    num1 = random.randrange(10)
    num2 = random.randrange(10)
    inp = input(str(num1) + ' * ' + str(num2) + ' = ')
    if int(inp) == num1*num2:
        print(':-)')
        cnt += 1
    else: print()
print("You got " + str(cnt) + " correct!")
print()

# Ex 8. Classes and objects
class fraction:
    numerator = 0
    denominator = 1

    def __init__(self, numer, denom):
        self.numerator = numer
        self.denominator = denom

    def __str__(self):
        return str(self.numerator) + "/" + str(self.denominator)

    def simplify(self):
        a, b = self.numerator, self.denominator
        while b:
            a, b = b, a%b
        self.numerator, self.denominator = int(self.numerator/a), int(self.denominator/a)
        

f = fraction(34, 62)

print(f)
f.simplify()
print(f)
