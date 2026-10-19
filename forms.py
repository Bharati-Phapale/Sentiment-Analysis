from django import forms
from .models import SentimentData

class SentimentDataForm(forms.ModelForm):
    RATING_CHOICES = [(i, str(i)) for i in range(1, 6)]  # Choices from 1 to 5
    
    rating = forms.ChoiceField(choices=RATING_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))

    class Meta:
        model = SentimentData
        fields = ['image', 'text', 'rating']
