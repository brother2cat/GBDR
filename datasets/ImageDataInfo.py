import torch


class mnistInfo(object):
    num_classes = 10
    channel = 1
    height = 28
    width = 28
    train_num = 60000
    test_num = 10000


class cifar10Info(object):
    num_classes = 10
    channel = 3
    height = 32
    width = 32
    train_num = 50000
    test_num = 10000
    label_class = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class cifar100Info(object):
    num_classes = 100
    channel = 3
    height = 32
    width = 32
    train_num = 50000
    test_num = 10000
    label_class = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class miniInfo(object):
    num_classes = 100
    channel = 3
    height = 64
    width = 64
    train_num = 48000
    test_num = 12000
    label_class = {'n01532829': 'house_finch',
                   'n01558993': 'robin',
                   'n01704323': 'triceratops',
                   'n01749939': 'green_mamba',
                   'n01770081': 'harvestman',
                   'n01843383': 'toucan',
                   'n01855672': 'goose',
                   'n01910747': 'jellyfish',
                   'n01930112': 'nematode',
                   'n01981276': 'king_crab',
                   'n02074367': 'dugong',
                   'n02089867': 'Walker_hound',
                   'n02091244': 'Ibizan_hound',
                   'n02091831': 'Saluki',
                   'n02099601': 'golden_retriever',
                   'n02101006': 'Gordon_setter',
                   'n02105505': 'komondor',
                   'n02108089': 'boxer',
                   'n02108551': 'Tibetan_mastiff',
                   'n02108915': 'French_bulldog',
                   'n02110063': 'malamute',
                   'n02110341': 'dalmatian',
                   'n02111277': 'Newfoundland',
                   'n02113712': 'miniature_poodle',
                   'n02114548': 'white_wolf',
                   'n02116738': 'African_hunting_dog',
                   'n02120079': 'Arctic_fox',
                   'n02129165': 'lion',
                   'n02138441': 'meerkat',
                   'n02165456': 'ladybug',
                   'n02174001': 'rhinoceros_beetle',
                   'n02219486': 'ant',
                   'n02443484': 'black-footed_ferret',
                   'n02457408': 'three-toed_sloth',
                   'n02606052': 'rock_beauty',
                   'n02687172': 'aircraft_carrier',
                   'n02747177': 'ashcan',
                   'n02795169': 'barrel',
                   'n02823428': 'beer_bottle',
                   'n02871525': 'bookshop',
                   'n02950826': 'cannon',
                   'n02966193': 'carousel',
                   'n02971356': 'carton',
                   'n02981792': 'catamaran',
                   'n03017168': 'chime',
                   'n03047690': 'clog',
                   'n03062245': 'cocktail_shaker',
                   'n03075370': 'combination_lock',
                   'n03127925': 'crate',
                   'n03146219': 'cuirass',
                   'n03207743': 'dishrag',
                   'n03220513': 'dome',
                   'n03272010': 'electric_guitar',
                   'n03337140': 'file',
                   'n03347037': 'fire_screen',
                   'n03400231': 'frying_pan',
                   'n03417042': 'garbage_truck',
                   'n03476684': 'hair_slide',
                   'n03527444': 'holster',
                   'n03535780': 'horizontal_bar',
                   'n03544143': 'hourglass',
                   'n03584254': 'iPod',
                   'n03676483': 'lipstick',
                   'n03770439': 'miniskirt',
                   'n03773504': 'missile',
                   'n03775546': 'mixing_bowl',
                   'n03838899': 'oboe',
                   'n03854065': 'organ',
                   'n03888605': 'parallel_bars',
                   'n03908618': 'pencil_box',
                   'n03924679': 'photocopier',
                   'n03980874': 'poncho',
                   'n03998194': 'prayer_rug',
                   'n04067472': 'reel',
                   'n04146614': 'school_bus',
                   'n04149813': 'scoreboard',
                   'n04243546': 'slot',
                   'n04251144': 'snorkel',
                   'n04258138': 'solar_dish',
                   'n04275548': 'spider_web',
                   'n04296562': 'stage',
                   'n04389033': 'tank',
                   'n04418357': 'theater_curtain',
                   'n04435653': 'tile_roof',
                   'n04443257': 'tobacco_shop',
                   'n04509417': 'unicycle',
                   'n04515003': 'upright',
                   'n04522168': 'vase',
                   'n04596742': 'wok',
                   'n04604644': 'worm_fence',
                   'n04612504': 'yawl',
                   'n06794110': 'street_sign',
                   'n07584110': 'consomme',
                   'n07613480': 'trifle',
                   'n07697537': 'hotdog',
                   'n07747607': 'orange',
                   'n09246464': 'cliff',
                   'n09256479': 'coral_reef',
                   'n13054560': 'bolete',
                   'n13133613': 'ear'}


if __name__ == '__main__':
    a = miniInfo()
    print(a.num_classes)
