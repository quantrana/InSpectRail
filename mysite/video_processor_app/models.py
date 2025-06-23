from django.db import models
from django.conf import settings # For ForeignKey to User
import os

class UploadedVideo(models.Model):
    """
    Stores uploaded video files and their processing status.
    """
    PROCESSING_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL, # Or models.CASCADE if videos should be deleted with user
        null=True,
        blank=True,
        help_text="User who uploaded the video."
    )
    file = models.FileField(
        upload_to='uploaded_videos/',
        help_text="The uploaded video file."
    )
    uploaded_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when the video was uploaded."
    )
    processing_status = models.CharField(
        max_length=20,
        choices=PROCESSING_STATUS_CHOICES,
        default='pending',
        help_text="Current status of video processing."
    )
    original_filename = models.CharField(
        max_length=255,
        blank=True,
        help_text="Original name of the uploaded file."
    )

    def __str__(self):
        return f"{self.original_filename or self.file.name} (Status: {self.get_processing_status_display()})"

    def save(self, *args, **kwargs):
        if self.file and not self.original_filename:
            self.original_filename = os.path.basename(self.file.name)
        super().save(*args, **kwargs)

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = "Uploaded Video"
        verbose_name_plural = "Uploaded Videos"


class VideoDetection(models.Model):
    """
    Stores a single defect detection within an uploaded video.
    """
    video = models.ForeignKey(
        UploadedVideo,
        related_name='detections',
        on_delete=models.CASCADE, # Detections are deleted if the video is deleted
        help_text="The video in which this detection was found."
    )
    timestamp_in_video = models.FloatField( # Storing as seconds (e.g., 123.45 seconds)
        help_text="Timestamp of the detection within the video, in seconds."
    )
    label = models.CharField(
        max_length=100,
        help_text="Type of defect detected (e.g., Crack, Squat)."
    )
    confidence = models.FloatField(
        help_text="Confidence score of the detection (0.0 to 1.0)."
    )
    # Storing bounding box as individual fields for easier querying if needed
    # Alternatively, could use a JSONField if database supports it well and complex queries aren't common
    bounding_box_x1 = models.IntegerField(help_text="Top-left X coordinate of the bounding box.")
    bounding_box_y1 = models.IntegerField(help_text="Top-left Y coordinate of the bounding box.")
    bounding_box_x2 = models.IntegerField(help_text="Bottom-right X coordinate of the bounding box.")
    bounding_box_y2 = models.IntegerField(help_text="Bottom-right Y coordinate of the bounding box.")

    detected_at = models.DateTimeField(
        auto_now_add=True, # Timestamp when this detection record was created
        help_text="Timestamp when this detection was recorded in the database."
    )

    def __str__(self):
        return f"{self.label} in {self.video.original_filename or self.video.file.name} at {self.timestamp_in_video:.2f}s (Conf: {self.confidence:.2f})"

    class Meta:
        ordering = ['video', 'timestamp_in_video']
        verbose_name = "Video Detection"
        verbose_name_plural = "Video Detections"
