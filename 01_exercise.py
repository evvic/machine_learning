# run py script through terminal with '/bin/python3'

# Ex 1. Print
print('Hello World')

# Ex 2. input, f-string
num1 = input("Enter first number: ")
num2 = input("Enter second number: ")
print(num1, ' * ', num2, ' = ', str(int(num1)*int(num2)))

# Ex 3. import, random
import random

num1 = random.randrange(100)
num2 = random.randrange(100)
print(str(num1), ' * ', str(num2), ' = ', str(num1*num2))

# Ex 4. loops
for i in range(0, 100, 10):
    print( [(j for j in range(1, 10, 1))] ,sep=' ')
# ^ unfinishes