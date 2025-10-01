num = int(input("Please enter the number for which you want to print table : "))
print("With for loop\n")
for i in range (1,11) :
    print(num,'*',i,'=',num*i)
print("With while loop")
i = 1
while i <= 10 :
    print(num,'*',i,'=',num*i)  
    i+=1