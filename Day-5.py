class Student:
    def __init__(self, name, mark1, mark2, mark3):
        self.name = name
        self.mark1 = int(mark1)
        self.mark2 = int(mark2)
        self.mark3 = int(mark3)

    def avgMarks(self):
        total = self.mark1 + self.mark2 + self.mark3
        print("Average Marks of ",self.name," = ",total/3)

student1 = Student("Shraddha", 78, 89,90)
student2 = Student("Khushi", 86, 78, 90)

student1.avgMarks()
student2.avgMarks()
    