from podcaster_graph.main import run

if __name__ == "__main__":
    result = run()
    print(result.get("audio_path", result))
