import requests
import random

review_samples = [
    "This movie was absolutely fantastic, I loved every scene.",
    "Terrible plot, wasted two hours of my life.",
    "One of the best films I've seen this year, incredible acting.",
    "Boring and predictable, I fell asleep halfway through.",
    "A masterpiece with stunning cinematography and a gripping story.",
    "Awful dialogue and wooden performances throughout.",
    "Heartwarming and beautifully shot, highly recommend.",
    "Complete waste of money, do not watch this.",
    "The soundtrack alone makes this worth watching.",
    "Poor pacing and a confusing storyline ruined it for me.",
    "I rewatched this last night and honestly, the real villain of the movie isn't Dr. Mann or the dust storms—it’s the sound mixing. I shouldn’t need subtitles to hear Matthew McConaughey whispering life-altering existential philosophy over Hans Zimmer absolutely slamming his hands on a pipe organ. Still cried like a baby during the video messages scene, though.",
    "It feels like a movie made by people who have had the concept of cinema explained to them once, second-hand, through a bad game of telephone. The dialogue is so stilted it feels like a placeholder script they forgot to rewrite before filming. I kind of love it for how spectacularly bizarre it is.",
    "Listen, say what you want about the plot, but the blue tint filter they used for this entire movie is an absolute vibe. It perfectly captures that specific, moody, rainy Pacific Northwest atmosphere. The baseball scene set to Muse is still unironically one of the greatest moments in cinematic history.",
    "The exact moment the original housekeeper rings the doorbell in the middle of a rainstorm is when this movie transitions from a brilliant, dark comedy into an absolute masterclass in tension. You think you know where it's going, and then the rug is pulled out so hard you get whiplash. Brilliant.",
    "This isn’t a movie, it’s just a recorded documentation of Adam Sandler paying his best friends millions of dollars to go on a lake house vacation with him. And honestly? Respect. It's comfort food cinema at its most low-effort, and sometimes that's exactly what you need on a lazy Sunday.",
    "Watching this movie feels like being trapped inside a panic attack for two hours straight. Adam Sandler’s character makes the absolute worst possible decision at every single crossroad, and you just want to scream at the screen. It's brilliant, but I can never watch it again.",
]

for i in range(35):
    text = random.choice(review_samples)
    response = requests.post(
        "http://localhost:8000/predict",
        json={"text": text}
    )
    print(i, response.status_code, response.json())


