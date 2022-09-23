from sanic import Sanic
from typing import Optional, Text
import src.config.constants as constants
import src.utils.loggers as loggers
import json


def create_app(confs: Optional[Text] = None):

    from src.utils.configs import Configuration
    Configuration.configurations = Configuration.read_config_file(confs + '/configurations.yml')
    loggers.get_out_log().info("configurations: {}.".format(json.dumps(Configuration.configurations)))

    from src.utils.loggers import configure_file_logging
    configure_file_logging(confs)

    app = Sanic(__name__)
    register_view(app)

    return app


def register_view(app):
    from src.view.view import setup_model, health_check, process_rec_info
    from src.model.model import BertModel

    app.modelSortLightGBM = BertModel()

    app.register_listener(setup_model, "before_server_start")
    # app.add_task() # 一些后台任务

    app.add_route(handler=health_check, uri="/", methods={"GET"})
    # get请求展示报错情况，日志如何记录。 post请求展示正常情况。
    app.add_route(handler=process_rec_info, uri="/recommendinfo", methods={"POST"})


def start_server(confs: Optional[Text] = None, port: int = constants.DEFAULT_SERVER_PORT):
    server = create_app(confs)
    protocol = "http"
    loggers.get_out_log().info(
        "Starting server on "
        "{}".format(constants.DEFAULT_SERVER_FORMAT.format(protocol, port))
    )
    server.run(host='0.0.0.0', port=port, debug=False, workers=1)


if __name__ == "__main__":
    start_server(confs='E:/Github/bert4torch/examples/serving/sanic_server/conf')
