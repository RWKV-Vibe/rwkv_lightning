echo -e "Testing FIM"

echo -e "\n\n========================================\n\n"
echo -e "\nDirect inference using the V1 interface\n"

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "✿prefix✿✿suffix✿感我此言良久立，却坐促弦弦转急。凄凄不似向前声，满座重闻皆掩泣。座中泣下谁最多？江州司马青衫湿。✿middle✿浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。",
      "✿prefix✿✿suffix✿感我此言良久立，却坐促弦弦转急。凄凄不似向前声，满座重闻皆掩泣。座中泣下谁最多？江州司马青衫湿。✿middle✿浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。",
      "✿prefix✿✿suffix✿感我此言良久立，却坐促弦弦转急。凄凄不似向前声，满座重闻皆掩泣。座中泣下谁最多？江州司马青衫湿。✿middle✿浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。"
    ],
    "max_tokens": 4096,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "noise": 0,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'

echo -e "\n\n========================================\n\n"
echo -e "\nBatch stream inference using [FIM/v1/batch-FIM interface]\n"

curl -X POST http://localhost:8000/FIM/v1/batch-FIM \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": [
      "浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。",
      "浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。",
      "浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。"
    ],
    "suffix": [
      "感我此言良久立，却坐促弦弦转急。凄凄不似向前声，满座重闻皆掩泣。座中泣下谁最多？江州司马青衫湿。",
      "感我此言良久立，却坐促弦弦转急。凄凄不似向前声，满座重闻皆掩泣。座中泣下谁最多？江州司马青衫湿。",
      "感我此言良久立，却坐促弦弦转急。凄凄不似向前声，满座重闻皆掩泣。座中泣下谁最多？江州司马青衫湿。"
    ],
    "max_tokens": 4096,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "noise": 0,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'

echo -e "\n\n========================================\n\n"
echo -e "\nBatch inference using [FIM/v1/batch-FIM interface]\n"

curl -X POST http://localhost:8000/FIM/v1/batch-FIM \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": [
      "浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。",
      "浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。",
      "浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。"
    ],
    "suffix": [
      "感我此言良久立，却坐促弦弦转急。凄凄不似向前声，满座重闻皆掩泣。座中泣下谁最多？江州司马青衫湿。",
      "感我此言良久立，却坐促弦弦转急。凄凄不似向前声，满座重闻皆掩泣。座中泣下谁最多？江州司马青衫湿。",
      "感我此言良久立，却坐促弦弦转急。凄凄不似向前声，满座重闻皆掩泣。座中泣下谁最多？江州司马青衫湿。"
    ],
    "max_tokens": 4096,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "noise": 0,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'

echo -e "\n\n========================================\n\n"
echo -e "\nBachSize == 1 Super Fast stream Infer with CUDA graph\n"

curl -X POST http://localhost:8000/FIM/v1/batch-FIM \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": [
      "浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。"
    ],
    "suffix": [
      "感我此言良久立，却坐促弦弦转急。凄凄不似向前声，满座重闻皆掩泣。座中泣下谁最多？江州司马青衫湿。"
    ],
    "stop_tokens": [0, 261, 24281],
    "max_tokens": 4096,
    "chunk_size": 64,
    "temperature": 0.8,
    "top_k": 1,
    "top_p": 0,
    "alpha_presence": 0,
    "alpha_frequency": 0,
    "alpha_decay": 0.996,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'

echo -e "\n\n========================================\n\n"
echo -e "\nBachSize == 1 Super Fast Infer with CUDA graph\n"

curl -X POST http://localhost:8000/FIM/v1/batch-FIM \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": [
      "浔阳江头夜送客，枫叶荻花秋瑟瑟。主人下马客在船，举酒欲饮无管弦。醉不成欢惨将别，别时茫茫江浸月。"
    ],
    "suffix": [
      "感我此言良久立，却坐促弦弦转急。凄凄不似向前声，满座重闻皆掩泣。座中泣下谁最多？江州司马青衫湿。"
    ],
    "max_tokens": 4096,
    "stop_tokens": [0, 261, 24281],
    "temperature": 0.8,
    "top_k": 1,
    "top_p": 0,
    "alpha_presence": 0,
    "alpha_frequency": 0,
    "alpha_decay": 0.996,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'