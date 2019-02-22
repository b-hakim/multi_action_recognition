import matplotlib.pyplot as plt


counts, labels = [], []

with open("num_videos_class.txt") as fr:
    for l in fr:
        count, cls = l.split()
        counts.append(int(count))
        # if cls == "Talking_with_human":
        #     cls = "Talking_with\nhuman"
        # elif cls == "Talking_in_phone":
        #     cls = "Talking_in\nphone"
        # elif cls == "Sleeping_on_desk":
        #     cls = "Sleeping_on\ndesk"
        # elif cls == "Shaking_hands":
        #     cls = "Shaking\nhands"
        labels.append(cls.replace("_", "\n"))


x_ticks = range(len(labels))

plt.bar(x_ticks, counts)
plt.xticks(x_ticks, labels, rotation=0, fontsize=10)
plt.ylabel("Nmber of videos")
plt.xlabel("Actions")
plt.title("Number of videos per action class")
plt.show()
