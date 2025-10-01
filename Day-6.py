class Shape:
    def area(self):
        pass

class Rectangle(Shape):
    def area(self, length, width):
        print(f'Area of Rectange with length of {length} and width of {width} is ',length*width)

class Circle(Shape):
    def area(self, radius):
        print(f'Area of Circle with radius {radius} is ', 22/7*radius)

radius = int(input("Enter the radius of circle for calculating the area : "))
length = int(input("Enter the length of the rectangle for calculting the area : "))
width = int(input("Enter the width of the rectangle for calculating the area : "))

circle = Circle()
rectangle = Rectangle()

circle.area(radius)
rectangle.area(length, width)
