import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import cv2
from deepface import DeepFace
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import lyricsgenius
import webbrowser
import random
import threading
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import concurrent.futures
import time
import os
import joblib
import pickle

#API KEYS
SPOTIPY_CLIENT_ID = ""
SPOTIPY_CLIENT_SECRET = ""
SPOTIPY_REDIRECT_URI = ""
GENIUS_ACCESS_TOKEN = ""

MAX_SONGS_TO_SCAN = 100
THREAD_COUNT = 10    
TRAINING_EPOCHS = 50    # Runs every time to refine accuracy

MODEL_FILE = "dj_brain_model.pkl"
VECT_FILE = "dj_brain_vect.pkl"
DATA_FILE = "dj_training_data.pkl"

WEIGHT_GENRE = 4   
WEIGHT_TITLE = 2   
WEIGHT_ARTIST = 3 
WEIGHT_LYRICS = 1  

training_data = [
    # ========================== HAPPY ==========================
    ("Happy Pop sunshine good times feel good", "happy"),
    ("Uptown Funk Funk dance jump saturday night", "happy"),
    ("Levitating Pop Disco glitter dance moonlight", "happy"),
    ("Walking on Sunshine Pop sunshine feel good", "happy"),
    ("Can't Stop the Feeling Pop dance body move joy", "happy"),
    ("Dynamite K-Pop light up like dynamite fun", "happy"),
    ("I Gotta Feeling Pop party good night tonight", "happy"),
    ("Shake It Off Pop shake play dance music", "happy"),
    ("Don't Stop Me Now Rock having a good time", "happy"),
    ("Dancing Queen Disco dance jive time of your life", "happy"),
    ("Shut Up and Dance Pop dance with me", "happy"),
    ("24K Magic Bruno Mars party gold fun", "happy"),
    ("Good Life happy sunshine smile", "happy"),
    ("Celebration celebrate good times", "happy"),
    ("Mundian To Bach Ke Bhangra party dance dhol beat", "happy"),
    ("Lamberghini Punjabi Pop drive gedi long drive", "happy"),
    ("Na Na Na Punjabi Pop dance party club nach", "happy"),
    ("Proper Patola Punjabi Pop swag beauty fashion", "happy"),
    ("Coka Punjabi dance party shava nach", "happy"),
    ("Yeah Baby Punjabi Pop smile happy vibe", "happy"),
    ("Diljit Dosanjh Pop Bhangra happy dance celebrate", "happy"),
    ("Bolo Tara Rara Daler Mehndi Bhangra fun", "happy"),
    ("Ishq Tera Guru Randhawa love romantic happy", "happy"),
    ("3 Peg Sharry Mann drink party bhangra fun", "happy"),
    ("High Rated Gabru Guru Randhawa swag style dance", "happy"),
    ("Daru Badnaam Kamal Kahlon party alcohol fun", "happy"),
    ("Illegal Weapon Jasmine Sandlas dance beat bhangra", "happy"),
    ("Siappa dance bhangra fun wedding", "happy"),
    ("ਨੱਚ ਭੰਗੜਾ ਖੁਸ਼ੀ ਪਿਆਰ", "happy"), 
    ("ਪੈਗ ਸ਼ਰਾਬ ਪਾਰਟੀ", "happy"),      
    ("ਜਸ਼ਨ ਮੁਬਾਰਕ", "happy"),          
    ("ਵਿਆਹ ਗਿੱਧਾ", "happy"),           
    ("ਮੌਜ ਮਸਤੀ", "happy"),             
    ("ਹੱਸਣਾ ਖੇਡਣਾ", "happy"),          
    ("ਸੋਹਣੀ ਕੁੜੀ", "happy"),           
    ("ਪਟੋਲਾ ਸਵੈਗ", "happy"),           
    ("Balam Pichkari Holi dance fun party masti", "happy"),
    ("London Thumakda Wedding dance dhol beat happy", "happy"),
    ("Aankh Marey Simmba dance party beat remix", "happy"),
    ("Kala Chashma Baar Baar Dekho dance wedding swag", "happy"),
    ("The Punjaabban Song dance party family fun", "happy"),
    ("Gallan Goodiyaan Dil Dhadakne Do dance family happy", "happy"),
    ("Abhi Toh Party Shuru Hui Hai Badshah club party dance", "happy"),
    ("Kar Gayi Chull Kapoor & Sons party dance fun", "happy"),
    ("Zingaat Dhadak energetic dance fast beat", "happy"),
    ("Badtameez Dil Yeh Jawaani Hai Deewani party dance fun", "happy"),
    ("Subha Hone Na De Desi Boyz party club dance", "happy"),
    ("Hookah Bar Khiladi 786 dance club beat", "happy"),
    ("नाच गाना मस्ती प्यार", "happy"), 
    ("खुशियां मनाओ", "happy"),         
    ("पार्टी डांस", "happy"),          
    ("प्यारा सुंदर", "happy"),         
    ("जिंदगी ना मिलेगी दोबारा", "happy"), 
    ("दिल धड़कने दो", "happy"),        
    ("خوشی ناچ گانا جشن", "happy"),   
    ("محبت پیار", "happy"),           
    ("خوبصورت زندگی", "happy"),       
    ("مسکراہٹ", "happy"),             
    ("Despacito Reggaeton dance playa party suave", "happy"),
    ("Vivir Mi Vida Salsa laugh live dance joy", "happy"),
    ("Danza Kuduro Reggaeton party hands up bailando", "happy"),
    ("Bailando Latin Pop dance street fire corazon", "happy"),
    ("Pepas Guaracha party club fiesta drugs fun", "happy"),
    ("Tití Me Preguntó Bad Bunny Dembow party girls", "happy"),
    ("Gasolina Reggaeton energy party motor", "happy"),
    ("Suavemente Elvis Crespo Merengue dance kiss", "happy"),
    ("Fiesta Loca", "happy"),
    ("Beso Amor", "happy"),
    ("Gangnam Style K-Pop dance funny energy sexy lady", "happy"),
    ("Boy With Luv BTS Pop love happy sunshine", "happy"),
    ("Cheer Up TWICE K-Pop cute happy energy", "happy"),
    ("Red Flavor Red Velvet K-Pop summer fruit fresh", "happy"),
    ("Fantastic Baby BIGBANG Dance party boom shakalaka", "happy"),
    ("Butter BTS Pop dance summer smooth", "happy"),
    ("사랑 행복 좋아 기쁨", "happy"),   
    ("춤 파티", "happy"),               
    ("재미 신나", "happy"),             
    ("웃음 미소", "happy"),             
    ("PonPonPon Kyary Pamyu Pamyu kawaii dance fun", "happy"),
    ("Koi Suru Fortune Cookie AKB48 happy dance idol", "happy"),
    ("Zenzenzense RADWIMPS upbeat energy run", "happy"),
    ("Peace Sign Kenshi Yonezu anime hero power happy", "happy"),
    ("Renai Circulation Kana Hanazawa cute love happy", "happy"),
    ("楽しい 嬉しい 笑顔 ダンス", "happy"), 
    ("幸せ 最高", "happy"),
    ("Shotta Flow (Remix) NLE Choppa Blueface Hype Bounce", "happy"),
    ("Outside (Better Days) Blueface OG Bobby Billions Anthemic West Coast", "happy"),
    ("Lucky You Eminem Joyner Lucas Technical Fast", "happy"),
    ("Praise The Lord (Da Shine) A$AP Rocky Skepta Classic Hype Flute", "happy"),
    ("Sprinter Dave Central Cee UK Drill Summer Hit", "happy"),
    ("Doja Central Cee Viral Catchy", "happy"),
    ("Surround Sound JID 21 Savage Baby Tate Trap Flow Switch", "happy"),
    ("Life Is Good Future Drake Switch-up Flex", "happy"),
    ("PUFFIN ON ZOOTIEZ Future Chill Trap Vibe", "happy"),
    ("Superhero (Heroes & Villains) Metro Boomin Future Chris Brown Cinematic Trap", "happy"),
    ("BOLO PENOMECO YDG K-Hip Hop Banger", "happy"),
    ("Vegas Doja Cat Pop Rap Catchy", "happy"),
    ("Dance Now JID Kenny Mason Fast Banger", "happy"),
    ("DANCE NOW Joey Valence & Brae Old School Fun", "happy"),
    ("Jimmy Cooks Drake 21 Savage Club Hard", "happy"),
    ("Rich Flex Drake 21 Savage Meme Hype", "happy"),
    ("Just Wanna Rock Lil Uzi Vert Jersey Club Dance", "happy"),
    ("N95 Kendrick Lamar Aggressive Bop", "happy"),
    ("Pure Cocaine Lil Baby Trap Hustle", "happy"),
    ("JASHAN-E-HIP-HOP Raftar Faris Shafi Desi Hip-Hop Bars", "happy"),
    ("Vossi Bop Stormzy UK Grime Viral", "happy"),
    ("Area Codes Kaliii Viral Fun", "happy"),
    ("Godzilla Eminem Juice WRLD Fast Rap World Record", "happy"),
    ("BTBT B.I Soulja Boy K-Pop Groove", "happy"),
    ("First Person Shooter Drake J. Cole Big 3 Flex", "happy"),
    ("Lovin On Me Jack Harlow Pop Rap Catchy", "happy"),
    ("Grey Yung Filly UK Afro-swing", "happy"),
    ("PAID Kanye West Ty Dolla $ign Club Bounce", "happy"),
    ("DO IT Kanye West Ty Dolla $ign Anthemic Hype", "happy"),
    ("HOTEL LOBBY Quavo Takeoff Trap Flex", "happy"),
    ("Lose Yourself Eminem Motivational Anthem", "happy"),
    ("All Of The Lights Kanye West Orchestral Anthem", "happy"),
    ("CARNIVAL Kanye West Ty Dolla $ign Chant Hype", "happy"),
    ("Goosebumps Travis Scott Psychedelic Hype", "happy"),
    ("Teach Me How to Dougie Cali Swag District Dance Throwback", "happy"),
    ("WHATS POPPIN Jack Harlow Club Catchy", "happy"),
    ("First Class Jack Harlow Sample Chill Flex", "happy"),
    ("Not Like Us Kendrick Lamar West Coast Bop", "happy"),
    ("The Box Roddy Ricch Viral Trap", "happy"),
    ("VOGUE PSK Drill Hype", "happy"),
    ("P-POP CULTURE Karan Aujla Ikky Anthemic Catchy", "happy"),
    ("SHUTDOWN Harkirat Sangha Energetic Flex", "happy"),
    ("Wealth Cheema Y Gur Sidhu Bhangra Beat Flex", "happy"),
    ("Champion's Anthem Karan Aujla Ikky Motivational Power", "happy"),
    ("CHOSEN ONE NIMAAN ARSH Confident Hype", "happy"),
    ("Hustle Baggh-e SMG Grind Motivation", "happy"),
    ("family ties Baby Keem Kendrick Lamar Explosive Switch-up", "happy"),
    ("squabble up Kendrick Lamar Rhythmic Bop", "happy"),
    ("Come On, Let's Go Tyler, The Creator Nigo Short Punchy", "happy"),
    ("Mary On A Cross Divine Deluxe Hitz Rock Catchy", "happy"),
    ("Replay Iyaz Throwback Catchy", "happy"),
    ("Candy Shop 50 Cent Club Flirty", "happy"),
    ("Cloud 9 Beach Bunny Happy Romantic", "happy"),
    ("Can I Call You Tonight? Dayglow Indie Pop Fun", "happy"),
    ("Best Friend Rex Orange County Sweet Indie", "happy"),
    ("Apple Cider beabadoobee Cute Crush", "happy"),
    ("we fell in love in october girl in red Indie Romantic", "happy"),
    ("Make It Right BTS Lauv Pop Hopeful", "happy"),
    ("At My Worst Andrew Foy Acoustic Cover Sweet", "happy"),
    ("Lovers Rock Ren Indie Groovy", "happy"),
    ("golden hour JVKE Magical Piano", "happy"),
    ("Just the Two of Us Fujii Kaze Jazz Cover Smooth", "happy"),
    ("Mr. Loverman Ricky Montgomery Crooner Indie", "happy"),
    ("The Way I Love You Michal Leah Ballad Wedding Vibe", "happy"),
    ("Good Looking Suki Waterhouse Slow Sultry", "happy"),
    ("It's Been a Long, Long Time Harry James Vintage Jazz", "happy"),
    ("City Of Stars Ryan Gosling Emma Stone Jazz Duet", "happy"),
    ("I Wanna Be Yours Arctic Monkeys Obsessive Slow", "happy"),
    ("I Want To Fall In Love Poeta Hambriento Spoken Romantic", "happy"),
    ("Zulfa Bir Daaku Romantic Vibe", "happy"),
    ("mona lisa mxmtoon Upbeat Indie", "happy"),
    ("Watermelon Sugar Harry Styles Summer Pop", "happy"),
    ("Blinding Lights The Weeknd Synth-wave Fast", "happy"),
    ("As It Was Harry Styles Indie Pop Fast", "happy"),
    ("Golden Harry Styles Bright Driving", "happy"),
    ("Crush Culture Conan Gray Anti-Love Pop Anthem", "happy"),
    ("Paramaniac Abby Roberts Pop Punk Energetic", "happy"),
    ("Mayonaka no Door Miki Matsubara City Pop Funk", "happy"),
    ("Idol YOASOBI Hyper-pop Fast", "happy"),
    ("Pompeii Bastille Choral Anthemic", "happy"),
    ("Honey, I'm Good. Andy Grammer Country Pop Happy", "happy"),
    ("Good bye-bye Ai Tomioka Upbeat Pop", "happy"),
    ("Good To Be Mark Ambor Feel-good Acoustic", "happy"),
    ("Everybody Talks Neon Trees Alt Rock Fun", "happy"),
    ("Electric Love BØRNS Glam Pop Anthemic", "happy"),
    ("Ho Hey The Lumineers Folk Stomp & Clap", "happy"),
    ("Don't Give Up On Me Andy Grammer Motivational Pop", "happy"),
    ("Into the I-LAND I-LAND K-Pop Anthem", "happy"),
    ("Bros Wolf Alice Indie Rock Friendship", "happy"),
    ("Say It Maggie Rogers Groovy Pop", "happy"),
    ("For A Reason Karan Aujla Ikky", "happy"),
    ("Boyfriend Karan Aujla Ikky", "happy"),
    ("Haseen Talwiinder NDS", "happy"),
    ("Luv Summer Umair Jevin Gill", "happy"),
    ("High On You Jind Universe", "happy"),
    ("Meri Zindagi Hai Tu Asim Azhar", "happy"),
    ("Sirra Guru Randhawa", "happy"),
    ("Cheques Shubh", "happy"),
    ("Make You Mine PUBLIC", "happy"),
    ("Golden Hour Fujii Kaze Remix JVKE", "happy"),
    ("Waka Waka (This Time for Africa) Shakira", "happy"),
    ("Wavin' Flag K'NAAN", "happy"),
    ("La La La (Brazil 2014) Shakira", "happy"),
    ("We Are One (Ole Ola) Pitbull J-Lo", "happy"),
    ("Magic in the Air Magic System", "happy"),
    ("Danza Kuduro Don Omar", "happy"),
    ("Pepas Farruko", "happy"),
    ("Shut Up and Dance WALK THE MOON", "happy"),
    ("I Gotta Feeling Black Eyed Peas", "happy"),
    ("We Are The Champions Queen", "happy"),
    ("World Cup IShowSpeed", "happy"),
    ("Ronaldinho Young Multi", "happy"),
    ("Peace Sign Kenshi Yonezu", "happy"),
    ("Blue Bird Ikimonogakari", "happy"),
    ("Silhouette KANA-BOON", "happy"),
    ("Go!!! FLOW", "happy"),
    ("Flyers BRADIO", "happy"),
    ("Lost In Paradise ALI AKLO", "happy"),
    ("Renai Circulation Monogatari Series", "happy"),
    ("Flyday Chinatown MerryGo", "happy"),
    ("B.O.M.B TREASURE", "happy"),
    ("Run TREASURE", "happy"),
    ("God's Menu Stray Kids", "happy"),
    ("Zenzenzense RADWIMPS", "happy"),
    ("Sparkle RADWIMPS", "happy"),
    ("Dream Lantern RADWIMPS", "happy"),
    ("Inferno Mrs. GREEN APPLE", "happy"),
    ("KUNOICHI BURNOUT SYNDROMES", "happy"),
    ("Butter BTS", "happy"),
    ("Dynamite BTS", "happy"),

    # ========================== SAD ==========================
    ("Someone Like You Soul Ballad never mind find someone like you cry", "sad"),
    ("Fix You Alt-Rock lights guide you home ignite bones", "sad"),
    ("Yesterday Classic Rock troubles seemed so far away", "sad"),
    ("Skinny Love Indie Folk breaking my heart skinny love", "sad"),
    ("The Sound of Silence Folk Rock darkness old friend silence", "sad"),
    ("Tears in Heaven Soft Rock know my name heaven tears", "sad"),
    ("Stay With Me Soul lonely morning sun stay", "sad"),
    ("Driver's License Pop Ballad cried drove alone suburbs", "sad"),
    ("Hurt Johnny Cash Country pain sting broken", "sad"),
    ("All I Want Kodaline crying leave me alone", "sad"),
    ("Let Her Go", "sad"),
    ("Say Something I'm giving up on you", "sad"),
    ("Qismat Punjabi Sad ro ro ke tears crying dhokha", "sad"),
    ("Mann Bharrya Punjabi Sad hurt pain chhod gaya", "sad"),
    ("Pachtaoge Arijit Singh Sad cheat betrayal cry", "sad"),
    ("Soch Hardy Sandhu Sad love pain distance", "sad"),
    ("Filhall Punjabi Sad broken heart memories miss you", "sad"),
    ("Judaai Punjabi emotional separation lonely", "sad"),
    ("Channa Mereya Sufi sad goodbye heart travel", "sad"),
    ("Daryaa Manmarziyaan sad love pain river", "sad"),
    ("Waalian harvest love soft sad cute", "sad"),
    ("Yaarr Ni Milyaa Hardy Sandhu sad betrayal", "sad"),
    ("ਦਿਲ ਟੁੱਟਿਆ ਰੋਣਾ ਦਰਦ", "sad"),   
    ("ਯਾਦ ਵਿਛੋੜਾ ਗਮ", "sad"),       
    ("ਧੋਖਾ ਬੇਵਫਾਈ", "sad"),         
    ("ਹੰਝੂ ਅੱਖੀਆਂ", "sad"),         
    ("ਇਕੱਲਾ ਉਦਾਸ", "sad"),          
    ("ਤਨਹਾਈ", "sad"),               
    ("Tum Hi Ho Aashiqui 2 sad love pain cry", "sad"),
    ("Ae Dil Hai Mushkil Arijit Singh sad unrequited love pain", "sad"),
    ("Kal Ho Naa Ho Sonu Nigam sad death goodbye cry", "sad"),
    ("Tujhe Bhula Diya Anjaana Anjaani sad heartbreak", "sad"),
    ("Kabira Yeh Jawaani Hai Deewani sad wedding goodbye", "sad"),
    ("Agar Tum Saath Ho Tamasha sad crying alka yagnik", "sad"),
    ("Bekhayali Kabir Singh sad angry pain scream", "sad"),
    ("Hamari Adhuri Kahani Arijit Singh sad incomplete love", "sad"),
    ("Judaai Badlapur sad separation pain", "sad"),
    ("Main Rahoon Ya Na Rahoon Armaan Malik sad memory death", "sad"),
    ("Tu Jaane Na Ajab Prem Ki Ghazab Kahani sad distance", "sad"),
    ("Deewane sad heartbreak longing love lost", "sad"),
    ("दर्द अकेला रोना जुदाई", "sad"), 
    ("गम आंसू दिल टूटा", "sad"),      
    ("बेवफा यादें", "sad"),           
    ("मौत बिछड़ना", "sad"),           
    ("अधूरी कहानी", "sad"),           
    ("गम دوری تنہائی اداس", "sad"),   
    ("آنسو دل ٹوٹا", "sad"),          
    ("جدائی", "sad"),                 
    ("بے وفائی", "sad"),              
    ("El Triste Ballad sad tears cry sorrow", "sad"),
    ("Corre Jesse & Joy Pop Ballad cry run away broken", "sad"),
    ("Recuérdame Coco Sad memory death miss you", "sad"),
    ("Historia de un Amor Bolero suffering pain love lost", "sad"),
    ("Amor Eterno Ranchera death crying miss you", "sad"),
    ("Tristeza Soledad", "sad"),
    ("Through the Night IU Ballad sad night miss you", "sad"),
    ("Spring Day BTS Ballad miss you snow wait", "sad"),
    ("Eyes, Nose, Lips Taeyang R&B regret memory sad", "sad"),
    ("If You BIGBANG Sad slow rain sorrow", "sad"),
    ("Breathe Lee Hi Ballad sigh tears comfort", "sad"),
    ("눈물 슬픔 아픔 이별", "sad"),   
    ("그리움 보고싶다", "sad"),       
    ("혼자 고독", "sad"),             
    ("Lemon Kenshi Yonezu sad death sorrow memory", "sad"),
    ("First Love Utada Hikaru heartbreak sad love", "sad"),
    ("One More Time, One More Chance sad lonely miss you", "sad"),
    ("Yuki no Hana Mika Nakashima snow sad love", "sad"),
    ("悲しい 涙 孤独 さよなら", "sad"), 
    ("会いたい 痛み", "sad"),
    ("Husn Anuv Jain", "sad"),
    ("Faasle Kaavish", "sad"),
    ("Tere Bin Nahi Lagda Asim Azhar", "sad"),
    ("Afsos Anuv Jain AP Dhillon", "sad"),
    ("Bikhra Abdul Hannan", "sad"),
    ("Saiyaara Tanishk Bagchi", "sad"),
    ("Pyaar Kyun Banaya Nehaal Naseem", "sad"),
    ("Chalo Door Kahin Samar Jafri", "sad"),
    ("Tere Pyar Main Kaavish", "sad"),
    ("Departure Lane Talha Anjum Umair", "sad"),
    ("Jo Tum Mere Ho Anuv Jain", "sad"),
    ("Nahin Milta Bayaan", "sad"),
    ("Samjho Na Aditya Rikhari", "sad"),
    ("Sahiba Aditya Rikhari", "sad"),
    ("Gal Sun Sabat Batin", "sad"),
    ("Heather Conan Gray", "sad"),
    ("Memories Conan Gray", "sad"),
    ("The One That Got Away Katy Perry", "sad"),
    ("Love In The Dark Adele", "sad"),
    ("Set Fire to the Rain Adele", "sad"),
    ("Someone To You BANNERS", "sad"),
    ("Space Song Beach House", "sad"),
    ("Glimpse of Us Joji", "sad"),
    ("Slow Dancing in the Dark Joji", "sad"),
    ("Mr. Loverman Ricky Montgomery", "sad"),
    ("Line Without a Hook Ricky Montgomery", "sad"),
    ("Another Love Tom Odell", "sad"),
    ("Atlantis Seafret", "sad"),
    ("Secret Base Anohana", "sad"),
    ("Nandemonaiya RADWIMPS", "sad"),
    ("Lemon Kenshi Yonezu", "sad"),
    ("Akuma no Ko Ai Higuchi", "sad"),
    ("Ghost In A Flower Akano", "sad"),
    ("One Day Omoinotake", "sad"),
    ("To everyone who want to die Takayan", "sad"),
    ("Killed by period pains and depression Takayan", "sad"),
    ("Just hide Takayan", "sad"),
    ("Wither Takayan", "sad"),
    ("Kokoronashi majiko", "sad"),
    ("Irony majiko", "sad"),
    ("Hated by life itself Raon", "sad"),
    ("Unravel TK Tokyo Ghoul", "sad"),

    # ========================== ANGRY ==========================
    ("Break Stuff Nu-Metal break stuff hate bad day", "angry"),
    ("Killing in the Name Metal Rap-Rock kill you do what they tell you", "angry"),
    ("Numb Rock tired of being what you want me to be", "angry"),
    ("In The End Rock Linkin Park tried so hard lose it all", "angry"),
    ("Hit 'Em Up Hip-Hop Rap kill enemy revenge hate", "angry"),
    ("Enter Sandman Metal nightmare sleep one eye open", "angry"),
    ("Humble Hip-Hop Rap sit down be humble punch", "angry"),
    ("Bodies Metal floor bodies hit the floor scream", "angry"),
    ("X Gon' Give It To Ya Rap fight punch aggressive", "angry"),
    ("Down with the Sickness Disturbed scream madness", "angry"),
    ("Monster Skillet", "angry"),
    ("Chop Suey System of a Down", "angry"),
    ("295 Sidhu Moose Wala Hip-Hop politics fight truth", "angry"),
    ("Levels Punjabi Rap attitude jatt fire gun", "angry"),
    ("Old Skool Prem Dhillon Gangster fight beat maar", "angry"),
    ("Same Beef Bohemia Rap fight dushmani hate", "angry"),
    ("Legend Sidhu Moose Wala Rap jatt fire killer", "angry"),
    ("Badmashi Punjabi Rap gangster fight revenge", "angry"),
    ("Goli AP Dhillon Rap gun shoot fight", "angry"),
    ("Syl Sidhu Moose Wala Rap politics rebel fight", "angry"),
    ("Last Ride Sidhu Moose Wala Rap death gangster", "angry"),
    ("Tochan Sidhu Moose Wala Rap attitude tractor fight", "angry"),
    ("Moosetape Sidhu Moose Wala Rap aggressive", "angry"),
    ("ਗੁੱਸਾ ਲੜਾਈ ਦੁਸ਼ਮਣ", "angry"),   
    ("ਬੰਦੂਕ ਗੋਲੀ ਮਾਰ", "angry"),      
    ("ਬਦਲਾ ਜੰਗ", "angry"),            
    ("ਹਥਿਆਰ ਖੂਨ", "angry"),           
    ("ਜੱਟ ਗੈਂਗਸਟਰ", "angry"),         
    ("ਲਲਕਾਰਾ", "angry"),              
    ("Apna Time Aayega Gully Boy Rap struggle fight anger", "angry"),
    ("Sadda Haq Rockstar Rock rebel fight anger", "angry"),
    ("Jee Karda Badlapur Rock aggressive scream pain", "angry"),
    ("Kar Har Maidan Fateh Sanju intense fight power", "angry"),
    ("Dhaakad Dangal aggressive fight wrestling strong", "angry"),
    ("Jai Ho Rock version aggressive win fight", "angry"),
    ("Bhaag D.K. Bose Delhi Belly Rock run aggressive", "angry"),
    ("Khalibali Padmaavat intense dark scary angry", "angry"),
    ("Aarambh Hai Prachand intense war fight speech", "angry"),
    ("Sher Aaya Gully Boy Rap aggressive attitude", "angry"),
    ("गुस्सा मार युद्ध नफरत", "angry"), 
    ("खून बदला", "angry"),            
    ("आक्रोश", "angry"),              
    ("दुश्मन", "angry"),              
    ("लड़ाई", "angry"),               
    ("غصہ نفرت لڑائی", "angry"),      
    ("جنگ بدلہ", "angry"),            
    ("دشمن خون", "angry"),            
    ("قتل", "angry"),                 
    ("Te Boté Trap hate ex-girlfriend leave me alone", "angry"),
    ("Ella Quiere Beber Anuel AA Trap aggressive club", "angry"),
    ("Mic Drop BTS Hip-Hop haters trophies fight", "angry"),
    ("Kill This Love Blackpink Pop intense break up kill", "angry"),
    ("God's Menu Stray Kids Hip-Hop cook aggressive shout", "angry"),
    ("Daechwita Agust D Rap king kill sword madness", "angry"),
    ("미친 싫어 죽어", "angry"),       
    ("화가 나", "angry"),               
    ("싸워 복수", "angry"),             
    ("Gurenge LiSA anime fight demon aggressive", "angry"),
    ("Unravel Tokyo Ghoul scream pain identity angry", "angry"),
    ("Megitsune Babymetal metal rock aggressive", "angry"),
    ("The Rumbling SiM titan fight war angry", "angry"),
    ("怒り 戦い 殺す 叫び", "angry"),   
    ("嫌い 敵", "angry"),               
    ("STFU AP Dhillon", "angry"),
    ("MF Gabhru! Karan Aujla", "angry"),
    ("Balenci Shubh", "angry"),
    ("Supreme Shubh", "angry"),
    ("Dhurandhar Hanumankind", "angry"),
    ("Shutup Call Talha Anjum", "angry"),
    ("Hit Em Up 2Pac", "angry"),
    ("Gangsta's Paradise Coolio", "angry"),
    ("get him back! Olivia Rodrigo", "angry"),
    ("Good 4 U Olivia Rodrigo", "angry"),
    ("Maniac Conan Gray", "angry"),
    ("Killing Me Conan Gray", "angry"),
    ("Jigsaw Conan Gray", "angry"),
    ("Winner Conan Gray", "angry"),
    ("Shinzo wo Sasageyo! Linked Horizon", "angry"),
    ("Guren no Yumiya Linked Horizon", "angry"),
    ("The Hero!! JAM Project", "angry"),
    ("Touch off UVERworld", "angry"),
    ("KICK BACK Kenshi Yonezu", "angry"),
    ("Ado Odo", "angry"),
    ("Rolling in the Deep Adele", "angry"),
    ("Guilty Conscience Tate McRae", "angry"),
    ("Run for the hills Tate McRae", "angry"),
    ("Look only at me Takayan", "angry"),
    ("Toy Takayan", "angry"),
    ("Life is a bitch! Takayan", "angry"),
    ("What's up, people?! MAXIMUM THE HORMONE", "angry"),
    ("MUKANJYO Survive Said The Prophet", "angry"),
    ("Paint It Black Rolling Stones", "angry"),

    # ========================== NEUTRAL ==========================
    ("Lo-Fi Beats Chillhop study relax background", "neutral"),
    ("Blue in Green Jazz Miles Davis trumpet calm", "neutral"),
    ("Weightless Ambient instrumental sleep relax", "neutral"),
    ("Intro Hip-Hop instrumental beginning start", "neutral"),
    ("Focus Flow Classical piano study quiet", "neutral"),
    ("Sunflower Pop Rap chill relaxing melody", "neutral"),
    ("Clair de Lune Classical piano calm moon", "neutral"),
    ("River Flows in You Yiruma Piano calm", "neutral"),
    ("Tum Se Hi Jab We Met Love romantic calm rain", "neutral"),
    ("Pee Loon Once Upon A Time Love romantic slow", "neutral"),
    ("Iktara Wake Up Sid Sufi calm soul", "neutral"),
    ("Kun Faya Kun Rockstar Sufi religious calm peace", "neutral"),
    ("Raabta Agent Vinod Love romantic slow piano", "neutral"),
    ("Tera Hone Laga Hoon Ajab Prem Ki Ghazab Kahani Love", "neutral"),
    ("Pani Da Rang Vicky Donor romantic guitar slow", "neutral"),
    ("सूफी सुकून शांति", "neutral"), 
    ("سکون", "neutral"),             
    ("Merry Christmas Mr Lawrence Ryuichi Sakamoto Piano", "neutral"),
    ("Summer Joe Hisaishi Piano calm relax", "neutral"),
    ("落ち着く 静か", "neutral"),
    ("Tu hai kahan AUR", "neutral"),
    ("Jhol Maanu", "neutral"),
    ("Savera Maanu", "neutral"),
    ("Wishes Hasan Raheem", "neutral"),
    ("FWM Hasan Raheem", "neutral"),
    ("Nashe Me Hun Hasan Raheem", "neutral"),
    ("Maand Bayaan", "neutral"),
    ("Aura Shubh", "neutral"),
    ("Wavy Karan Aujla", "neutral"),
    ("Dil Nu AP Dhillon", "neutral"),
    ("Without Me AP Dhillon", "neutral"),
    ("Khayaal Talwiinder", "neutral"),
    ("Nasha Talwiinder", "neutral"),
    ("Regardless Asim Azhar", "neutral"),
    ("Sukoon Hassan & Roshaan", "neutral"),
    ("Memories Hasan Raheem", "neutral"),
    ("Sweater Weather The Neighbourhood", "neutral"),
    ("Sex, Drugs, Etc. Beach Weather", "neutral"),
    ("Apocalypse Cigarettes After Sex", "neutral"),
    ("Cigarette Daydreams Cage The Elephant", "neutral"),
    ("death bed (coffee for your head) Powfu", "neutral"),
    ("Night Dancer imase", "neutral"),
    ("Mayonaka no Door Stay With Me Miki Matsubara", "neutral"),
    ("Summertime Maggie", "neutral"),
    ("Plastic Palm Trees Tate McRae", "neutral"),
    ("Ditto NewJeans", "neutral"),
    ("Off The Record IVE", "neutral"),
    ("Suzume RADWIMPS", "neutral"),
    ("The Name of Life Spirited Away", "neutral"),
    ("Path of the Wind Totoro", "neutral"),
    ("Tokyo Wonder. Nakimushi", "neutral"),
    ("Yakuza Lofi", "neutral"),
    ("Count What You Have Now Vantage", "neutral"),
    ("Stan Eminem Dido Storytelling Dark", "neutral"),
    ("No Role Modelz J. Cole Classic Anthem", "neutral"),
    ("Wet Dreamz J. Cole Storytelling Nostalgic", "neutral"),
    ("Money Trees Kendrick Lamar Classic Vibe", "neutral"),
    ("HUMBLE. Kendrick Lamar Impact Bounce", "neutral"),
    ("Obsessed With You Central Cee Sample Drill Love", "neutral"),
    ("She Knows J. Cole Melodic Vibe", "neutral"),
    ("My Band D12 Comedy Pop Rap", "neutral"),
    ("SIXPACK Wes Patrick Indie Rap Flow", "neutral"),
    ("Still Smokin Snoop Dogg West Coast Chill", "neutral"),
    ("Big Poppa The Notorious B.I.G. Classic Smooth", "neutral"),
    ("What's Beef? The Notorious B.I.G. Dark Story", "neutral"),
    ("Snow On Tha Bluff J. Cole Conscious Slow", "neutral"),
    ("Change J. Cole Inspirational Lyrical", "neutral"),
    ("Without Me Eminem Comedy Classic", "neutral"),
    ("Black Friday Kendr Cole Remix Bars", "neutral"),
    ("t h e . c l i m b . b a c k J. Cole Lyrical Complex", "neutral"),
    ("Zeus Eminem Lyrical Apology", "neutral"),
    ("From The D 2 The LBC Eminem Snoop Dogg West Coast Fast", "neutral"),
    ("Still D.R.E. Dr. Dre Snoop Dogg Classic West Coast", "neutral"),
    ("meet the grahams Kendrick Lamar Dark Diss", "neutral"),
    ("Mona Lisa Lil Wayne Kendrick Lamar Storytelling Masterpiece", "neutral"),
    ("Many Men (Wish Death) 50 Cent Classic Gangsta", "neutral"),
    ("Gangsta's Paradise Coolio Classic Choir", "neutral"),
    ("HiiiPower Kendrick Lamar Conscious Revolutionary", "neutral"),
    ("Alright Kendrick Lamar Anthemic Hopeful", "neutral"),
    ("4:44 JAY-Z Apologetic Reflective", "neutral"),
    ("The Way Life Goes Lil Uzi Vert Melodic Sad", "neutral"),
    ("All The Way Live Metro Boomin Future Lil Uzi Vert Soundtrack Vibe", "neutral"),
    ("All The Stars Kendrick Lamar SZA Cinematic Pop Rap", "neutral"),
    ("Calling Metro Boomin Swae Lee Melodic Soundtrack", "neutral"),
    ("LET GO Central Cee Sample Drill Emotional", "neutral"),
    ("Losing Interest - Remix Stract Shiloh Dynasty Lo-fi Sad", "neutral"),
    ("STORYBOOK / EVERYTHING Pertinence Indie Chill", "neutral"),
    ("TIL FURTHER NOTICE Travis Scott James Blake 21 Savage Dark Atmospheric", "neutral"),
    ("See You Again Tyler, The Creator Kali Uchis Dreamy Romantic", "neutral"),
    ("KEYS TO MY LIFE Kanye West Ty Dolla $ign Melodic Synth", "neutral"),
    ("Under The Sun Dreamville J. Cole Soulful Intro", "neutral"),
    ("bon iver mxmtoon Acoustic Chill", "neutral"),
    ("Juliet Cavetown Indie Chill", "neutral"),
    ("Head In The Clouds Adiescar Chase Ambient Soundtrack", "neutral"),
    ("People Watching Conan Gray Observational Mid-tempo", "neutral"),
    ("Strawberries & Cigarettes Troye Sivan Nostalgic Pop", "neutral"),
    ("thinkin of you Keno Carter ay3demi R&B Smooth", "neutral"),
    ("close with desires Teo Glacier R&B Chill", "neutral"),
    ("rises the moon Liana Flores Folk Lullaby", "neutral"),
    ("L.A. will hyde tiffi dress Chill Pop", "neutral"),
    ("Hallucinogenics - Stripped Matt Maeson Acoustic Raw", "neutral"),
    ("NIGHT DANCER imase Groovy J-Pop", "neutral"),
    ("Babydoll Dominic Fike Indie Short", "neutral"),
    ("Somewhere Only We Know Keane Nostalgic Piano Rock", "neutral"),
    ("My old story IU K-Pop Ballad", "neutral"),
    ("Rich Heart Gurinder Gill Manu Smooth Melodic", "neutral"),
    ("savera Maanu Turhan James Indie Pop Chill", "neutral"),
    ("At Peace Karan Aujla Ikky Reflective Laid back", "neutral"),
    ("I Wonder Kanye West Soulful Inspiring", "neutral"),
    ("luther Kendrick Lamar SZA Melodic Soft", "neutral"),
    ("Farewell, Neverland TOMORROW X TOGETHER Pop Rock Dramatic", "neutral"),
    ("Back To Me The Rose Indie Rock Emotional", "neutral"),
    ("505 Arctic Monkeys Indie Rock Dramatic", "neutral"),
    ("Notion The Rare Occasions Indie Rock Fast", "neutral"),
    ("Skyfall Adele Orchestral Bond Theme", "neutral"),
]

mood_buckets = {"happy": [], "sad": [], "angry": [], "neutral": []}

class AIDJApp:
    def __init__(self, root, window_title="Ultimate AI DJ"):
        self.root = root
        self.root.title(window_title)
        self.root.geometry("1000x850")
        self.root.configure(bg="#1E1E1E")

        # ui
        self.header = tk.Label(root, text="⚡ Turbo AI DJ", font=("Helvetica", 24, "bold"), bg="#1E1E1E", fg="#1DB954")
        self.header.pack(pady=10)

        self.video_frame = tk.Label(root, bg="black")
        self.video_frame.pack(pady=5)

        self.stats_label = tk.Label(root, text="Songs Analyzed: 0", font=("Arial", 12), bg="#1E1E1E", fg="white")
        self.stats_label.pack(pady=5)

        self.mood_label = tk.Label(root, text="DETECTED MOOD: ...", font=("Helvetica", 16, "bold"), bg="#1E1E1E", fg="yellow")
        self.mood_label.pack(pady=10)

        # controls
        self.btn_frame = tk.Frame(root, bg="#1E1E1E")
        self.btn_frame.pack(pady=5)

        self.scan_btn = tk.Button(self.btn_frame, text="1. Scan Full Library", command=self.start_scanning, font=("Arial", 12), bg="#535353", fg="white", width=30)
        self.scan_btn.pack(side=tk.LEFT, padx=10)

        self.play_btn = tk.Button(self.btn_frame, text="2. Match Mood & Play", command=self.play_music, font=("Arial", 12), bg="#1DB954", fg="white", width=30, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=10)

        # learning and feedback
        self.feedback_frame = tk.LabelFrame(root, text="AI Feedback / Correction", bg="#1E1E1E", fg="white", font=("Arial", 10))
        self.feedback_frame.pack(pady=15, padx=20, fill="x")
        
        self.lbl_feedback = tk.Label(self.feedback_frame, text="Was the last song wrong? Teach the AI:", bg="#1E1E1E", fg="white")
        self.lbl_feedback.pack(pady=5)

        self.btn_fix_happy = tk.Button(self.feedback_frame, text="Actually HAPPY", command=lambda: self.correct_mistake("happy"), bg="#FFD700", width=20)
        self.btn_fix_happy.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_fix_sad = tk.Button(self.feedback_frame, text="Actually SAD", command=lambda: self.correct_mistake("sad"), bg="#1E90FF", width=20)
        self.btn_fix_sad.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_fix_angry = tk.Button(self.feedback_frame, text="Actually ANGRY", command=lambda: self.correct_mistake("angry"), bg="#FF4500", width=20)
        self.btn_fix_angry.pack(side=tk.LEFT, padx=10, pady=10)

        self.log_box = scrolledtext.ScrolledText(root, height=12, width=110, bg="black", fg="#00FF00", font=("Consolas", 9))
        self.log_box.pack(pady=10, padx=10)

        self.cap = cv2.VideoCapture(0)
        self.current_mood = "neutral"
        self.ai_model = None
        self.sp = None
        self.genius = None
        self.is_scanning = False
        self.total_songs_found = 0
        
        self.last_played_song = None
        self.last_played_features = None
        self.training_data = list(training_data) # load from const

        self.check_saved_data()

        self.update_video()

    def log(self, message):
        """Thread-safe logging to GUI"""
        self.root.after(0, self._log_internal, message)

    def _log_internal(self, message):
        self.log_box.insert(tk.END, f"> {message}\n")
        self.log_box.see(tk.END)

    def check_saved_data(self):
        if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE) and os.path.exists(DATA_FILE):
            self.log("📂 Found existing AI Brain! Loading...")
            try:
                self.classifier = joblib.load(MODEL_FILE)
                self.vectorizer = joblib.load(VECT_FILE)
                with open(DATA_FILE, 'rb') as f:
                    self.training_data = pickle.load(f)
                self.log(f"Loaded Brain trained on {len(self.training_data)} examples.")
                self.ai_model_loaded = True
            except Exception as e:
                self.log(f"Error loading files: {e}. Will retrain.")
                self.ai_model_loaded = False
        else:
            self.log("🆕 No saved brain found. AI will train fresh on first scan.")
            self.ai_model_loaded = False

    def save_brain(self):
        try:
            joblib.dump(self.classifier, MODEL_FILE)
            joblib.dump(self.vectorizer, VECT_FILE)
            with open(DATA_FILE, 'wb') as f:
                pickle.dump(self.training_data, f)
            self.log("AI Brain Saved Successfully.")
        except Exception as e:
            self.log(f"Could not save brain: {e}")

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                self.current_mood = analysis[0]['dominant_emotion']
                x, y, w, h = analysis[0]['region']['x'], analysis[0]['region']['y'], analysis[0]['region']['w'], analysis[0]['region']['h']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            except:
                pass

            self.mood_label.config(text=f"DETECTED MOOD: {self.current_mood.upper()}")
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.root.after(10, self.update_video)

    def start_scanning(self):
        if self.is_scanning: return
        self.is_scanning = True
        self.scan_btn.config(state=tk.DISABLED)
        self.log("Starting High-Speed Scan...")
        threading.Thread(target=self.run_backend_logic).start()

    def run_backend_logic(self):
        self.log("🧠 Refining AI Model (Incremental Learning)...")
        
        X = [item[0] for item in self.training_data]
        y = [item[1] for item in self.training_data]
        classes = np.unique(y)

        # 1. Prepare Vectorizer
        if hasattr(self, 'vectorizer'):
            # If we loaded a brain, we MUST use the existing vocabulary to match the classifier weights
            # Warning: New words in the massive dataset that weren't in the saved file will be ignored here.
            # This is the trade-off for "not retraining from scratch".
            X_vectors = self.vectorizer.transform(X)
        else:
            # Fresh start
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            X_vectors = self.vectorizer.fit_transform(X)

        # 2. Prepare Classifier
        if not hasattr(self, 'classifier'):
            self.classifier = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, random_state=42, learning_rate='adaptive', eta0=0.05)

        # 3. Train! (Always runs now)
        X_train, X_val, y_train, y_val = train_test_split(X_vectors, y, test_size=0.1, random_state=5) 
        
        for epoch in range(1, TRAINING_EPOCHS + 1):
            self.classifier.partial_fit(X_train, y_train, classes=classes)
            
            # Metrics
            train_preds = self.classifier.predict(X_train)
            train_probs = self.classifier.predict_proba(X_train)
            train_acc = accuracy_score(y_train, train_preds)
            train_loss = log_loss(y_train, train_probs, labels=classes)

            val_preds = self.classifier.predict(X_val)
            val_probs = self.classifier.predict_proba(X_val) 
            val_acc = accuracy_score(y_val, val_preds)
            val_loss = log_loss(y_val, val_probs, labels=classes)
            
            msg = f"   🔄 Epoch {epoch}/{TRAINING_EPOCHS} - Train Acc: {train_acc*100:.1f}% Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.1f}% Loss: {val_loss:.4f}"
            self.log(msg)
            print(msg) 
            time.sleep(0.01) 

        self.log("AI Model Updated & Saved.")
        self.save_brain()

        self.log("🔑 Connecting to APIs...")
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=SPOTIPY_CLIENT_ID,
                client_secret=SPOTIPY_CLIENT_SECRET,
                redirect_uri=SPOTIPY_REDIRECT_URI,
                scope="user-library-read user-read-private",
                open_browser=False
            ))
            self.genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, verbose=False, timeout=5) 
            self.log("Connected.")
        except Exception as e:
            self.log(f"Error: {e}")
            return

        self.log(f"Fetching song list (Limit: {MAX_SONGS_TO_SCAN})...")
        
        all_tracks = []
        offset = 0
        batch_size = 50
        
        while True:
            try:
                if len(all_tracks) >= MAX_SONGS_TO_SCAN: break
                results = self.sp.current_user_saved_tracks(limit=batch_size, offset=offset)
                items = results['items']
                if not items: break
                all_tracks.extend(items)
                offset += batch_size
                self.log(f"   Fetched {len(all_tracks)} songs so far...")
            except Exception as e:
                self.log(f"Fetch Error: {e}")
                break

        # BATCH FETCH GENRES
        self.log(f"🎵 Fetching Genres for {len(all_tracks)} songs...")
        artist_ids = set()
        for item in all_tracks:
            if item['track'] and item['track']['artists']:
                artist_ids.add(item['track']['artists'][0]['id'])
        
        artist_genres = {}
        artist_ids_list = list(artist_ids)
        for i in range(0, len(artist_ids_list), 50):
            chunk = artist_ids_list[i:i + 50]
            try:
                artists_data = self.sp.artists(chunk)
                for artist in artists_data['artists']:
                    artist_genres[artist['id']] = " ".join(artist['genres'])
            except:
                pass

        self.log(f"🚀 Starting WEIGHTED AI Analysis using {THREAD_COUNT} threads...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
            future_to_song = {
                executor.submit(self.analyze_single_track, item, artist_genres): item 
                for item in all_tracks
            }
            for future in concurrent.futures.as_completed(future_to_song):
                try:
                    result = future.result()
                    if result:
                        mood, info, features = result
                        mood_buckets[mood].append({"info": info, "features": features})
                        self.total_songs_found += 1
                        self.update_stats()
                        self.log(f"   [AI DECISION] {info['name']} -> {mood.upper()}")
                except:
                    pass

        if not mood_buckets['sad']: mood_buckets['sad'] = mood_buckets['neutral']
        if not mood_buckets['happy']: mood_buckets['happy'] = mood_buckets['neutral']
        
        self.log("Scan Complete!")
        self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL, bg="#1ED760"))

    def analyze_single_track(self, track_item, artist_genres_map):
        track = track_item['track']
        if not track: return None
        
        name = track['name']
        artist_obj = track['artists'][0]
        artist = artist_obj['name']
        artist_id = artist_obj['id']
        
        genre_text = artist_genres_map.get(artist_id, "")

        lyrics_text = ""
        try:
            song_lyrics = self.genius.search_song(name, artist)
            if song_lyrics and song_lyrics.lyrics:
                lyrics_text = song_lyrics.lyrics[:250].replace("\n", " ")
        except:
            pass

        # Weighted Input
        features = (f"{name} " * WEIGHT_TITLE) + \
                   (f"{artist} " * WEIGHT_ARTIST) + \
                   (f"{genre_text} " * WEIGHT_GENRE) + \
                   (f"{lyrics_text} " * WEIGHT_LYRICS)
        
        vectorized_input = self.vectorizer.transform([features])
        mood = self.classifier.predict(vectorized_input)[0]
        
        return (mood, {"name": name, "artist": artist, "url": track['external_urls']['spotify']}, features)

    def update_stats(self):
        text = f"Happy: {len(mood_buckets['happy'])} | Sad: {len(mood_buckets['sad'])} | Angry: {len(mood_buckets['angry'])} | Total: {self.total_songs_found}"
        self.root.after(0, lambda: self.stats_label.config(text=text))

    def play_music(self):
        emotion = self.current_mood
        target = "neutral"
        if emotion in ["happy", "surprise"]: target = "happy"
        elif emotion in ["sad", "fear"]: target = "sad"
        elif emotion in ["angry", "disgust"]: target = "angry"
        
        self.log(f"🤖 User is {emotion}. Playing {target.upper()} music...")
        
        songs = mood_buckets[target]
        if not songs:
            if not mood_buckets['sad']: mood_buckets['sad'] = mood_buckets['neutral']
            if not mood_buckets['happy']: mood_buckets['happy'] = mood_buckets['neutral']
            songs = mood_buckets.get(target, [])

        if songs:
            selection = random.choice(songs)
            choice = selection["info"]
            self.last_played_song = selection
            self.last_played_features = selection["features"]
            self.log(f"🎵 Opening: {choice['name']} - {choice['artist']}")
            webbrowser.open(choice['url'])
        else:
            self.log("⚠️ Library empty! Playing random fallback.")
            all_songs = sum(mood_buckets.values(), [])
            if all_songs: webbrowser.open(random.choice(all_songs)["info"]['url'])

    def correct_mistake(self, correct_mood):
        if not self.last_played_song:
            self.log("⚠️ Play a song first before correcting!")
            return

        name = self.last_played_song["info"]["name"]
        self.log(f"🎓 TEACHING AI: '{name}' is actually {correct_mood.upper()}...")
        
        self.training_data.append((self.last_played_features, correct_mood))
        
        new_X_vec = self.vectorizer.transform([self.last_played_features])
        self.classifier.partial_fit(new_X_vec, [correct_mood])
        self.save_brain()
        
        for bucket in mood_buckets.values():
            if self.last_played_song in bucket:
                bucket.remove(self.last_played_song)
        
        mood_buckets[correct_mood].append(self.last_played_song)
        self.update_stats()
        self.log("AI Brain Updated & Saved!")

if __name__ == "__main__":
    root = tk.Tk()
    app = AIDJApp(root)

    root.mainloop()
