from django import forms
from .models import UploadedVideo

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedVideo
        fields = ['file'] # Only allow uploading the file itself through this form
        widgets = {
            'file': forms.ClearableFileInput(attrs={'accept': 'video/*'}),
        }
        labels = {
            'file': 'Select video file to upload',
        }
        help_texts = {
            'file': 'Supported formats: MP4, AVI, MOV, etc. (Depends on server capabilities)',
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['file'].required = True
        # You can add more validation here if needed, e.g., file size, specific content types.
        # For example, to limit file size (this is a basic client-side hint, server-side validation is more robust):
        # self.fields['file'].widget.attrs.update({'data-max-size': '50000000'}) # 50MB example
