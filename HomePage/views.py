from django.shortcuts import render

def HomePage(request):
    return render(request, 'HomePage/HomePage.html')