""" 
Copyright @ 2025 Ethan W. Han
Computer Science & Software Engineering Dept.
Miami University OH
"""

from ollama import Client
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def call_chat(host_name, model_name, prompt, image_path=None):
    """
    Function to send prompts to models
    from Ollama.

    host (str): Link to the server containing the models.
    model (str): Name of the model.
    prompt (str): Prompt input to the model.
    image_path (str): Path to the image input.
    """

    # Note: Connecting to the host requires VPN.
    client = Client(
    host=host_name,
    )

    # Cases for whether an image path is given.
    response = None
    if image_path == None:
        response = client.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
            }]
        )
    else:
        response = client.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }]
        )
    return response['message']['content']


def plot_image(image_path, title=None):
    """
    Plots an image using matplotlib.

    image_path (str): Path to the image file.
    title (str): Optional title for the image plot.
    """
    # Load the image
    img = mpimg.imread(image_path)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    if title:
        plt.title(title)
    plt.show()
