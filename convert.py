# Replace this with the old file name, don't include the extension (.txt)
old_file_name = "NLCE_Resum9_ss_U22"

# Open the old file and read it line by line
old_file = open(old_file_name + ".txt", 'r').readlines()
new_lines = []
# Adjust the old lines by clearing the empty spaces at the start and end
for line in old_file:
    new_lines.append(line.strip() + "\n")

# Write the data to the new file
new_file = open(old_file_name + "_new.txt", 'w')
# Exclude the first two lines that label the data
new_file.writelines(new_lines[2:])
new_file.close()

