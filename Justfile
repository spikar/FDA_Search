image_name := "fda-search"
container_name := "fda-search"

rebuild: rm-container rm-image build
 
rm-container:
  docker rm -f {{container_name}}
 
rm-image:
  docker rmi -f {{image_name}}
 
build: rm-container rm-image
  docker build -t {{image_name}} .
 
run: build
  docker run -d --network genai_network -p 5001:8000 --restart unless-stopped --name {{container_name}} -it {{image_name}}
 
watch: run
  docker logs -f {{container_name}}
