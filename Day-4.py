num = int(input("Enter the number to check weather it is prime or not : "))
file = open("Prime.txt", "w")
isPrime = True

for i in range(2, num//2):
    if num % i == 0:
        isPrime = False

if isPrime:
    file.write(f"{num} is a prime number")
else:
    file.write(f"{num} is not a prime number")

file.close()