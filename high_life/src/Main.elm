module Main exposing (..)

import Browser
import Html exposing (..)
import Html.Attributes exposing (style)
import Html.Events exposing (..)
import Http
import Json.Decode as Decode exposing (Decoder, decodeString, float, int, nullable, string)
import Json.Decode.Pipeline exposing (hardcoded, optional, required)
import Json.Encode
import RemoteData exposing (RemoteData(..), WebData)
import RemoteData.Http
import Url.Builder exposing (crossOrigin)



-- MAIN


main =
    Browser.element
        { init = init
        , update = update
        , subscriptions = subscriptions
        , view = view
        }



-- MODEL


type alias Model =
    { input : String
    , request : WebData LLMRequest
    , llmResponse : List LLMResponse
    }


type alias LLMRequest =
    { text : String
    }


type alias LLMResponse =
    { text : String
    }


init : () -> ( Model, Cmd Msg )
init _ =
    ( Model "" NotAsked [], Cmd.none )



-- UPDATE


type Msg
    = SendRequest String
    | UpdateRequest String
    | UseResponseToUpdateModel (WebData LLMResponse)


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        SendRequest input ->
            ( { model | request = Loading }, postLLMRequest input )

        UpdateRequest newRequest ->
            ( { model | input = newRequest }, Cmd.none )

        UseResponseToUpdateModel webDataResponse ->
            case webDataResponse of
                Success new_llmResponse ->
                    ( { input = "", request = NotAsked, llmResponse = new_llmResponse :: model.llmResponse }, Cmd.none )

                NotAsked ->
                    ( model, Cmd.none )

                Failure err ->
                    ( { model | request = Failure err }, Cmd.none )

                Loading ->
                    ( { model | request = Loading }, Cmd.none )



-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions model =
    Sub.none



-- VIEW


view : Model -> Html Msg
view model =
    div []
        [ h2 []
            [ text "Recommender"
            , h3 [] [ text "Get recommendations on Food, Wine, Travel, and more from trusted sources." ]
            ]
        , viewQuote model
        ]


viewResponses : Model -> Html Msg
viewResponses model =
    let
        x y =
            tr []
                [ td [] [ text "input" ]
                , td [] [ text y.text ]
                ]
    in
    --div []
    table []
        [ thead []
            [ th [] [ text "Query" ]
            , th [] [ text "Answer" ]
            ]
        , tbody [] <| List.map x model.llmResponse
        ]


viewQuote : Model -> Html Msg
viewQuote model =
    let
        state =
            model.request
    in
    case state of
        NotAsked ->
            div []
                [ text "Try a search"
                , input [ onInput UpdateRequest ] []
                , button [ onClick (SendRequest model.input) ] [ text "Ask!" ]
                , viewResponses model
                ]

        Failure err ->
            decodeError model err

        Loading ->
            div []
                [ text "Loading..."
                , viewResponses model
                ]

        Success _ ->
            div []
                [ text "Try a search"
                , input [ onInput UpdateRequest ] []
                , button [ onClick <| SendRequest model.input ] [ text "Ask!" ]
                , viewResponses model
                ]



-- HTTP


postLLMRequest : String -> Cmd Msg
postLLMRequest input =
    --  (crossOrigin "http://localhost:8001/query" [] [])
    RemoteData.Http.postWithConfig RemoteData.Http.defaultConfig "/query" UseResponseToUpdateModel llmResponseDecoder (encodeRequest input)


encodeRequest : String -> Json.Encode.Value
encodeRequest input =
    Json.Encode.object
        [ ( "text", Json.Encode.string input ) ]


llmResponseDecoder : Decoder LLMResponse
llmResponseDecoder =
    Decode.succeed LLMResponse
        |> required "text" string


decodeError : Model -> Http.Error -> Html Msg
decodeError model error =
    case error of
        Http.BadUrl string ->
            div []
                [ text string
                , viewResponses model
                ]

        Http.Timeout ->
            div []
                [ text "Timeout :("
                , viewResponses model
                ]

        Http.NetworkError ->
            div []
                [ text "Network Error :("
                , viewResponses model
                ]

        Http.BadStatus int ->
            div []
                [ text ("Bad Status: " ++ String.fromInt int)
                , viewResponses model
                ]

        Http.BadBody string ->
            div []
                [ text ("Bad Request Body :(" ++ " " ++ string)
                , viewResponses model
                ]
