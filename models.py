from django.db import models

class SentimentData(models.Model):
    RATING_CHOICES = [(i, str(i)) for i in range(1, 6)]  # Choices from 1 to 5

    image = models.ImageField(upload_to='uploads/')  # Stores images in 'media/uploads/'
    text = models.TextField()
    rating = models.IntegerField(choices=RATING_CHOICES, default=3)  # Default rating is 3
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Sentiment Data {self.id} - Rating: {self.rating}"
