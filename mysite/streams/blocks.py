"""Streamfields are here"""
from wagtail import blocks
from wagtail.images.blocks import ImageChooserBlock
from wagtail.embeds.blocks import EmbedBlock

class TitleAndTextBlock(blocks.StructBlock):
    """Title and text only"""

    title = blocks.CharBlock(required=False, help_text = 'Add your title')
    text = blocks.TextBlock(required=False, help_text = 'Add additional text')

    class Meta:
        template = "streams/title_and_text_block.html"
        icon = "edit"
        label = "Title & Text"

class CardBlock(blocks.StructBlock):
    """Cards with image and text and button(s)"""
    title = blocks.CharBlock(required=False, help_text = 'Add your title')

    cards = blocks.ListBlock(
        blocks.StructBlock(
            [
                ("image", ImageChooserBlock(required=False)),
                ("title", blocks.CharBlock(required=False, max_length=40)),
                ("text", blocks.TextBlock(required=False, max_length=200)),
                ("button_page", blocks.PageChooserBlock(required=False)),
                ("button_url", blocks.URLBlock(required=False, help_text="If the button page above is selected, that will be prioritised")),
            ]
        )
    )

    class Meta:
        template = "streams/card_block.html"
        icon = "placeholder"
        label = "Cards"



class RichtextBlock(blocks.RichTextBlock):
    """Richtext with all features"""

    class Meta:
        template = "streams/richtext_block.html"
        icon = "doc-full"
        label = "Full RichText"



class SimpleRichtextBlock(blocks.RichTextBlock):
    """Simple Rich Text  with limited features"""

    def __init__(self, required=True, help_text=None, editor='default', features=None, **kwargs):
        super().__init__(**kwargs)
        self.features = [
            "bold",
            "italic",
            "link",
        ]


    class Meta:
        template = "streams/richtext_block.html"
        icon = "edit"
        label = "Simple RichText"


from wagtail import blocks
from wagtail.images.blocks import ImageChooserBlock

class HorizontalListItemBlock(blocks.StructBlock):
    """Single item in a horizontal list: image and caption text"""
    image = ImageChooserBlock(required=True)
    text = blocks.CharBlock(required=True, max_length=100)

    class Meta:
        label = "Horizontal List Item"

class HorizontalListBlock(blocks.StructBlock):
    """A horizontal list of image-text items"""
    title = blocks.CharBlock(required=False, help_text="Optional title for the list")

    items = blocks.ListBlock(
        HorizontalListItemBlock(),
        label="List Items"
    )

    class Meta:
        template = "streams/blocks/horizontal_list_block.html"
        icon = "list-ul"
        label = "Horizontal List"
    
class VideoBlock(blocks.StructBlock):
    """Embedded video with optional caption"""
    video_url = EmbedBlock(required=True)
    caption = blocks.CharBlock(required=False)
    class Meta:
        template = "streams/video_block.html"
        icon = "media"
        label = "Embedded Video"
