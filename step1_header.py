import os

PLY = "data/truck_30000.ply"
header_lines = []
with open(PLY, "rb") as f:
    while True:
        line = f.readline().decode("ascii").strip()
        header_lines.append(line)
        if line == "end_header":
            break
    header_bytes = f.tell()

print("HEADER : ")
for l in header_lines:
    print(l)
n = 0
props = []
for l in header_lines:
    if l.startswith("element vertex"):
        n = int(l.split()[-1])
    if l.startswith("property float"):
        props.append(l.split()[-1])

file_size = os.path.getsize(PLY)
bytes_per_g = len(props) * 4

print("\nOverview :")
print(f"Number of Gaussians N : {n:,}")
print(f"Properties / Gaussian : {len(props)}")
print(f"Bytes / Gaussian      : {bytes_per_g}")
print(f"Header                : {header_bytes} bytes")
print(f"Expected size         : {header_bytes + n * bytes_per_g:,} bytes")
print(f"Actual size           : {file_size:,} bytes")
print(f"=> {file_size / 1024**2:.1f} MB")
