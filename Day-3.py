students = {
    "Shraddha" : 96,
    "Khushi" : 94,
    "Devyanshi" : 82,
    "Gupta" : 93
}

ans = ""

for i in students:
    if(ans == "" or students[ans] < students[i]):
        ans = i

print("Highest Marks is scored by ",ans)