from predict import Predictor

if __name__ == '__main__':
    animation_prompts = '''0: masterpiece, a man wearing a cap taking a picture with a camera in a city street with buildings in the background and a mural painted on the ground, Chris LaBrooy, anamorphic, graffiti art, video art, Full HD, vibrant colors, dynamic lighting, ultra high detail, dramatic lighting, movie, poster, asymmetric composition, ultra detailed, Photorealistic, unreal engine, art by Johnson Ting--neg
| 30: masterpiece, Traveler taking photos with a camera, metropolis, kuala lumpur city skyline, kuala lumpur buildings in the background, beach and palms, airbus airplane, bright colorful, photo realistic, Natural Lighting, dynamic lighting, ultra high detail, dramatic lighting, movie, poster, asymmetric composition, ultra detailed, Photorealistic, unreal engine, art by Johnson Ting--neg' 
| 60: masterpiece, Traveler taking photos with a camera, metropolis, beach and palm trees, airbus airplane, bright colorful, photo realistic, words, Boston, New York
'''
    cog_predictor = Predictor()
    cog_predictor.setup()
    output = cog_predictor.predict(model_checkpoint='Protogen_V2.2.ckpt',
                               #use_init=True,
                               #init_image='https://t3.ftcdn.net/jpg/04/55/47/34/360_F_455473421_Kjo4MjABmDuL7MQYkElJ0D9KdtSZxqAV.jpg',
                               max_frames=60,
                               animation_prompts=animation_prompts,
                               sampler="dpm2")
    print(output)
